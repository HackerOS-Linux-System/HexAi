use crate::{
    auth::login,
    static_gui::serve_gui,
    engine::{EngineKind, LlmEngine, Mode},
    rag::chunk_text,
    state::AppState,
    tools::{run_tool_loop_stream, run_tool_loop_sync},
};
use axum::{
    body::Body,
    extract::{Multipart, Path, Query, State},
    http::{HeaderValue, Method, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{convert::Infallible, sync::Arc};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::{cors::CorsLayer, limit::RequestBodyLimitLayer};
use uuid::Uuid;

// ── DTOs ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatReq {
    pub session_id: Option<String>,
    pub message:    String,
    #[serde(default)]
    pub stream:     bool,
}

#[derive(Serialize)]
pub struct ChatResp { pub session_id: String, pub response: String }

#[derive(Deserialize)] pub struct EngineReq { pub engine: String }
#[derive(Deserialize)] pub struct ModeReq   { pub mode:   String }
#[derive(Deserialize)] pub struct RagReq    { pub query:  String }
#[derive(Deserialize)] pub struct ImageReq  { pub prompt: String }
#[derive(Deserialize)] pub struct TtsQuery  { pub text:   String }
#[derive(Deserialize)] pub struct IngestReq {
    pub source:                        String,
    #[serde(default = "d500")] pub chunk_size:    usize,
    #[serde(default = "d100")] pub chunk_overlap: usize,
}
fn d500() -> usize { 500 }
fn d100() -> usize { 100 }

// ── Router ────────────────────────────────────────────────────────

pub fn build_router(state: AppState) -> Router {
    let cors       = build_cors(&state.cfg.cors_origins);
    let auth_state = Arc::clone(&state.auth);

    let public = Router::new()
        .route("/health",     get(health))
        .route("/auth/token", post(login).with_state(auth_state.clone()))
        // Serve embedded Next.js GUI – /gui and /gui/*
        .route("/gui",        get(gui_index))
        .route("/gui/",       get(gui_index))
        .route("/gui/*path",  get(gui_static))
        // Root redirect to /gui
        .route("/",           get(gui_index));

    // Each protected handler is a plain async fn – no captures, fully Send.
    let protected = Router::new()
        .route("/chat",           post(chat_handler))
        .route("/engine",         post(set_engine))
        .route("/mode",           post(set_mode))
        .route("/rag",            post(rag_query))
        .route("/ingest",         post(ingest_doc))
        .route("/generate_image", post(generate_image))
        .route("/transcribe",     post(transcribe))
        .route("/analyze_image",  post(analyze_image))
        .route("/tts",            get(tts))
        .route("/stats",          get(stats))
        .route("/sessions",       get(list_sessions))
        .route("/session/:id",    delete(delete_session))
        .layer(middleware::from_fn_with_state(auth_state, crate::auth::auth_middleware))
        .with_state(state.clone());

    Router::new()
        .merge(public)
        .merge(protected)
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024))
        .layer(cors)
        .with_state(state)
}

fn build_cors(origins: &[String]) -> CorsLayer {
    if origins.iter().any(|o| o == "*") {
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
            .allow_headers(tower_http::cors::Any)
    } else {
        let parsed: Vec<HeaderValue> = origins.iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(parsed)
            .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
            .allow_headers(tower_http::cors::Any)
    }
}

// ── Sanitize ──────────────────────────────────────────────────────

fn sanitize(prompt: &str) -> String {
    let re1 = regex::Regex::new(r"(?i)(zignoruj|ignore|forget).*?instrukcje").unwrap();
    let re2 = regex::Regex::new(r"(?i)(podaj|give|show).*(hasło|password|secret|token)").unwrap();
    re2.replace_all(&re1.replace_all(prompt, "[FILTERED]"), "[FILTERED]")
        .trim()
        .to_string()
}

// ── Helper: build enriched prompt (all awaits up front, no locks held) ──

struct ChatContext {
    enriched: String,
    history:  Vec<(String, String)>,
    engine:   LlmEngine,
    serper:   Option<String>,
}

async fn build_chat_context(st: &AppState, raw_prompt: &str, session_id: &str) -> ChatContext {
    // 1. RAG – await completes before we touch engine lock
    let rag_hits = st.rag.search(raw_prompt, 3).await;

    // 2. User facts (sync, no await)
    let facts = st.profiler.get_facts(session_id, 5);

    // 3. History (awaited)
    let history = st.memory.get_trimmed_history(session_id).await;

    // 4. Engine clone (lock taken and immediately dropped)
    let engine = st.engine.read().await.clone();

    // 5. Serper key (cheap clone of Option<String>)
    let serper = st.cfg.serper_api_key.clone();

    // Build enriched prompt
    let mut enriched = raw_prompt.to_string();
    if !rag_hits.is_empty() {
        enriched = format!("{enriched}\n\nKontekst z bazy wiedzy:\n{}", rag_hits.join("\n---\n"));
    }
    if !facts.is_empty() {
        enriched = format!("Fakty o użytkowniku: {}\n\n{enriched}", facts.join(", "));
    }

    ChatContext { enriched, history, engine, serper }
}

// ── /chat ─────────────────────────────────────────────────────────

async fn chat_handler(
    State(st): State<AppState>,
    Json(req): Json<ChatReq>,
) -> Response {
    let session_id = req.session_id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
    let prompt     = sanitize(&req.message);

    let ctx = build_chat_context(&st, &prompt, &session_id).await;

    if req.stream {
        // Everything needed by the spawn is owned/cloned – fully 'static + Send
        let st2     = st.clone();
        let sid     = session_id.clone();
        let p       = prompt.clone();
        let enr     = ctx.enriched.clone();
        let hist    = ctx.history.clone();
        let eng     = ctx.engine.clone();
        let serper  = ctx.serper.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<String>(512);

        tokio::spawn(async move {
            run_tool_loop_stream(&eng, &enr, &hist, serper.as_deref(), tx).await;
            // Persist: re-run sync so we have a clean string to store
            if let Ok(saved) = eng.generate_sync(&enr, &hist).await {
                st2.memory.add_message(&sid, &p, &saved).await;
                st2.profiler.update_from_message(&sid, &p);
            }
        });

        let stream = ReceiverStream::new(rx).map(Ok::<_, Infallible>);
        Response::new(Body::from_stream(stream))
    } else {
        match run_tool_loop_sync(&ctx.engine, &ctx.enriched, &ctx.history, ctx.serper.as_deref()).await {
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR,
                       Json(json!({"error": e.to_string()}))).into_response(),
            Ok(response) => {
                st.memory.add_message(&session_id, &prompt, &response).await;
                st.profiler.update_from_message(&session_id, &prompt);
                Json(ChatResp { session_id, response }).into_response()
            }
        }
    }
}

// ── /engine  /mode ────────────────────────────────────────────────

async fn set_engine(
    State(st): State<AppState>,
    Json(req): Json<EngineReq>,
) -> impl IntoResponse {
    let kind = match req.engine.as_str() {
        "ollama" => EngineKind::Ollama,
        "openai" => EngineKind::OpenAI,
        "candle" => EngineKind::Candle,
        other    => return (StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("Nieznany silnik: {other}")}))).into_response(),
    };
    st.set_engine_kind(kind).await;
    Json(json!({"message": format!("Silnik: {}", req.engine)})).into_response()
}

async fn set_mode(
    State(st): State<AppState>,
    Json(req): Json<ModeReq>,
) -> impl IntoResponse {
    let mode = match req.mode.as_str() {
        "general"     => Mode::General,
        "programista" => Mode::Programista,
        other         => return (StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("Nieznany tryb: {other}")}))).into_response(),
    };
    st.set_mode(mode).await;
    Json(json!({"message": format!("Tryb: {}", req.mode)})).into_response()
}

// ── /rag ──────────────────────────────────────────────────────────

async fn rag_query(
    State(st): State<AppState>,
    Json(req): Json<RagReq>,
) -> impl IntoResponse {
    // All awaits happen before we call into_response()
    let hits   = st.rag.search(&req.query, 3).await;
    let engine = st.engine.read().await.clone();  // clone, lock dropped immediately
    drop(st);

    let context = if hits.is_empty() {
        "Brak dokumentów w bazie wiedzy.".to_string()
    } else {
        hits.join("\n---\n")
    };
    let n      = hits.len();
    let prompt = format!(
        "Odpowiedz na pytanie na podstawie kontekstu:\n\n{context}\n\nPytanie: {}",
        req.query
    );

    match engine.generate_sync(&prompt, &[]).await {
        Ok(r)  => Json(json!({"response": r, "sources": n})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR,
                   Json(json!({"error": e.to_string()}))).into_response(),
    }
}

// ── /ingest ───────────────────────────────────────────────────────

async fn ingest_doc(
    State(st): State<AppState>,
    Json(req): Json<IngestReq>,
) -> impl IntoResponse {
    match fetch_text(&req.source).await {
        Err(e) =>
            (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
        Ok(t) if t.trim().is_empty() =>
            (StatusCode::BAD_REQUEST, Json(json!({"error": "Empty document"}))).into_response(),
        Ok(t) => {
            let chunks = chunk_text(&t, req.chunk_size, req.chunk_overlap, &req.source);
            let n      = chunks.len();
            st.rag.add_documents(chunks).await;
            Json(json!({
                "message": format!("Zindeksowano {n} fragmentów z {}", req.source),
                "chunks":  n
            })).into_response()
        }
    }
}

async fn fetch_text(source: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if source.starts_with("http://") || source.starts_with("https://") {
        let bytes = reqwest::get(source).await?.bytes().await?;
        if let Ok(text) = pdf_extract::extract_text_from_mem(&bytes) {
            if !text.trim().is_empty() { return Ok(text); }
        }
        let html = String::from_utf8_lossy(&bytes).to_string();
        let doc  = scraper::Html::parse_document(&html);
        let sel  = scraper::Selector::parse("body").map_err(|e| format!("{e:?}"))?;
        Ok(doc.select(&sel).flat_map(|e| e.text()).collect::<Vec<_>>().join(" "))
    } else {
        if !std::path::Path::new(source).exists() {
            return Err(format!("Plik {source} nie istnieje.").into());
        }
        if source.ends_with(".pdf") {
            Ok(pdf_extract::extract_text(source)?)
        } else {
            Ok(tokio::fs::read_to_string(source).await?)
        }
    }
}

// ── Stubs ─────────────────────────────────────────────────────────

async fn generate_image(_: Json<ImageReq>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED,
     Json(json!({"error": "Image generation: start the Python service"})))
}
async fn transcribe(_: Multipart) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED,
     Json(json!({"error": "Transcription: start the Python service"})))
}
async fn analyze_image(_: Multipart) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED,
     Json(json!({"error": "Vision: start the Python service"})))
}
async fn tts(_: Query<TtsQuery>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, "TTS: start the Python service")
}

// ── Stats / Sessions / Health ─────────────────────────────────────

async fn stats(State(st): State<AppState>) -> impl IntoResponse {
    let (kind, mode) = {
        let eng = st.engine.read().await;
        (eng.kind.to_string(), eng.mode.to_string())
    };
    let sessions = st.memory.list_sessions().await;
    Json(json!({
        "engine":             kind,
        "mode":               mode,
        "vram_used_gb":       null,
        "vram_total_gb":      null,
        "active_sessions":    sessions.len(),
        "history_len":        0,
        "model_loaded":       true,
        "model_idle_seconds": 0.0,
        "rag_chunks":         st.rag.count(),
        "auth_enabled":       st.auth.auth_enabled,
    }))
}

async fn list_sessions(State(st): State<AppState>) -> impl IntoResponse {
    Json(json!({"sessions": st.memory.list_sessions().await}))
}

async fn delete_session(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    st.memory.clear_session(&id).await;
    Json(json!({"message": format!("Sesja {id} usunięta")}))
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok", "version": "2.0.0"}))
}

// ── GUI static file handlers ──────────────────────────────────────

async fn gui_index() -> Response {
    serve_gui("index.html").await
}

async fn gui_static(axum::extract::Path(path): axum::extract::Path<String>) -> Response {
    serve_gui(&path).await
}
