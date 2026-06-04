use crate::{
    engine::{Engine, Mode},
    error::{HexError, Result},
    rag::chunk_text,
    state::AppState,
    tools::dispatch_tool,
};
use axum::{
    body::Body,
    extract::{Multipart, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
    routing::{delete, get, post},
    Json, Router,
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, convert::Infallible};
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

// ── Request / Response types ─────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatReq {
    pub session_id: Option<String>,
    pub message: String,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Serialize)]
pub struct ChatResp {
    pub session_id: String,
    pub response: String,
}

#[derive(Deserialize)]
pub struct EngineReq { pub engine: String }

#[derive(Deserialize)]
pub struct ModeReq { pub mode: String }

#[derive(Deserialize)]
pub struct RagReq { pub query: String }

#[derive(Deserialize)]
pub struct IngestReq {
    pub source: String,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_overlap")]
    pub chunk_overlap: usize,
}
fn default_chunk_size() -> usize { 500 }
fn default_overlap() -> usize { 100 }

#[derive(Deserialize)]
pub struct ImageReq { pub prompt: String }

#[derive(Deserialize)]
pub struct TtsQuery { pub text: String }

// ── Router builder ────────────────────────────────────────────────

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/chat",          post(chat_handler))
        .route("/engine",        post(set_engine))
        .route("/mode",          post(set_mode))
        .route("/rag",           post(rag_query))
        .route("/ingest",        post(ingest_doc))
        .route("/generate_image",post(generate_image))
        .route("/transcribe",    post(transcribe))
        .route("/analyze_image", post(analyze_image))
        .route("/tts",           get(tts))
        .route("/stats",         get(stats))
        .route("/sessions",      get(list_sessions))
        .route("/session/:id",   delete(delete_session))
        .route("/health",        get(health))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state)
}

// ── Sanitize ──────────────────────────────────────────────────────

fn sanitize(prompt: &str) -> String {
    let re1 = regex::Regex::new(r"(?i)(zignoruj|ignore|forget|przestań|nie\s+słuchaj).*?instrukcje").unwrap();
    let re2 = regex::Regex::new(r"(?i)(podaj|give|show).*(hasło|password|secret|token|klucz|key)").unwrap();
    let s = re1.replace_all(prompt, "[FILTERED]");
    re2.replace_all(&s, "[FILTERED]").trim().to_string()
}

// ── Tool loop helpers ─────────────────────────────────────────────

fn parse_tool_calls(text: &str) -> Vec<(String, HashMap<String, String>)> {
    let re = regex::Regex::new(r"\{tool:(\w+)\s+(.+?)\}").unwrap();
    let arg_re = regex::Regex::new(r#"(\w+)="([^"]+)""#).unwrap();
    re.captures_iter(text)
        .map(|cap| {
            let name = cap[1].to_string();
            let args = arg_re.captures_iter(&cap[2])
                .map(|a| (a[1].to_string(), a[2].to_string()))
                .collect();
            (name, args)
        })
        .collect()
}

// ── Chat handler ──────────────────────────────────────────────────

async fn chat_handler(
    State(st): State<AppState>,
    Json(req): Json<ChatReq>,
) -> Response {
    let session_id = req.session_id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
    let prompt = sanitize(&req.message);

    // Enrich with RAG context
    let mut enriched = prompt.clone();
    let rag_hits = st.rag.search(&prompt, 3);
    if !rag_hits.is_empty() {
        enriched = format!("{enriched}\n\nKontekst:\n{}", rag_hits.join("\n"));
    }

    // User facts
    let facts = st.profiler.get_facts(&session_id, 5);
    if !facts.is_empty() {
        enriched = format!("Fakty o użytkowniku: {}\n\n{enriched}", facts.join(", "));
    }

    let history: Vec<(String, String)> = st.memory
        .get_history(&session_id)
        .await
        .into_iter()
        .map(|t| (t.user, t.assistant))
        .collect();

    if req.stream {
        let st2 = st.clone();
        let sid = session_id.clone();
        let eng = st.engine.read().await;
        let stream_result = eng.generate_stream(&enriched, &history).await;
        drop(eng);

        match stream_result {
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
            Ok(token_stream) => {
                let (tx, rx) = tokio::sync::mpsc::channel::<String>(256);
                tokio::spawn(async move {
                    let mut full = String::new();
                    tokio::pin!(token_stream);
                    while let Some(token) = token_stream.next().await {
                        match token {
                            Ok(t) => {
                                full.push_str(&t);
                                let _ = tx.send(t).await;
                            }
                            Err(_) => break,
                        }
                    }
                    st2.memory.add_message(&sid, &prompt, &full).await;
                    st2.profiler.update_from_message(&sid, &prompt);
                });

                let rx_stream = ReceiverStream::new(rx)
                    .map(|t| Ok::<_, Infallible>(t));
                Response::new(Body::from_stream(rx_stream))
            }
        }
    } else {
        let eng = st.engine.read().await;
        match eng.generate_sync(&enriched, &history).await {
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
            Ok(response) => {
                drop(eng);
                st.memory.add_message(&session_id, &prompt, &response).await;
                st.profiler.update_from_message(&session_id, &prompt);
                Json(ChatResp { session_id, response }).into_response()
            }
        }
    }
}

// ── Engine / Mode ─────────────────────────────────────────────────

async fn set_engine(State(st): State<AppState>, Json(req): Json<EngineReq>) -> impl IntoResponse {
    let kind = match req.engine.as_str() {
        "transformers" => Engine::Transformers,
        "ollama"       => Engine::Ollama,
        other          => return (StatusCode::BAD_REQUEST, Json(json!({"error": format("Nieznany silnik: {other}")}))).into_response(),
    };
    st.set_engine_kind(kind).await;
    Json(json!({"message": format!("Silnik: {}", req.engine)})).into_response()
}

async fn set_mode(State(st): State<AppState>, Json(req): Json<ModeReq>) -> impl IntoResponse {
    let mode = match req.mode.as_str() {
        "general"     => Mode::General,
        "programista" => Mode::Programista,
        other         => return (StatusCode::BAD_REQUEST, Json(json!({"error": format("Nieznany tryb: {other}")}))).into_response(),
    };
    st.set_mode(mode).await;
    Json(json!({"message": format!("Tryb: {}", req.mode)})).into_response()
}

// ── RAG ───────────────────────────────────────────────────────────

async fn rag_query(State(st): State<AppState>, Json(req): Json<RagReq>) -> impl IntoResponse {
    let hits = st.rag.search(&req.query, 3);
    let context = if hits.is_empty() { "Brak dokumentów.".into() } else { hits.join("\n") };
    let prompt = format!("Odpowiedz na pytanie na podstawie kontekstu:\n\n{context}\n\nPytanie: {}", req.query);
    let eng = st.engine.read().await;
    match eng.generate_sync(&prompt, &[]).await {
        Ok(r)  => Json(json!({"response": r})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Ingest ────────────────────────────────────────────────────────

async fn ingest_doc(State(st): State<AppState>, Json(req): Json<IngestReq>) -> impl IntoResponse {
    let text = fetch_text(&req.source).await;
    match text {
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
        Ok(t) if t.trim().is_empty() => (StatusCode::BAD_REQUEST, Json(json!({"error": "Empty document"}))).into_response(),
        Ok(t) => {
            let chunks = chunk_text(&t, req.chunk_size, req.chunk_overlap, &req.source);
            let n = chunks.len();
            st.rag.add_documents(chunks);
            Json(json!({"message": format!("Zindeksowano {n} fragmentów z {}", req.source)})).into_response()
        }
    }
}

async fn fetch_text(source: &str) -> anyhow::Result<String> {
    if source.starts_with("http://") || source.starts_with("https://") {
        let bytes = reqwest::get(source).await?.bytes().await?;
        // Try PDF
        if let Ok(text) = pdf_extract::extract_text_from_mem(&bytes) {
            return Ok(text);
        }
        let html = String::from_utf8_lossy(&bytes).to_string();
        let doc = scraper::Html::parse_document(&html);
        let sel = scraper::Selector::parse("body").unwrap();
        Ok(doc.select(&sel).flat_map(|e| e.text()).collect::<Vec<_>>().join(" "))
    } else {
        if !std::path::Path::new(source).exists() {
            anyhow::bail!("Plik {source} nie istnieje.");
        }
        if source.ends_with(".pdf") {
            Ok(pdf_extract::extract_text(source)?)
        } else {
            Ok(tokio::fs::read_to_string(source).await?)
        }
    }
}

// ── Stubs for vision / TTS / Whisper / image-gen ─────────────────
// These require heavy Python deps; expose endpoints that return clear errors.

async fn generate_image(Json(req): Json<ImageReq>) -> impl IntoResponse {
    Json(json!({"error": "Image generation requires the Python service. Start with: hexai --with-python"}))
}

async fn transcribe(mut multipart: Multipart) -> impl IntoResponse {
    Json(json!({"error": "Transcription requires the Python service."}))
}

async fn analyze_image(mut multipart: Multipart) -> impl IntoResponse {
    Json(json!({"error": "Vision analysis requires the Python service."}))
}

async fn tts(Query(q): Query<TtsQuery>) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, "TTS requires the Python service.")
}

// ── Stats / Session mgmt / Health ────────────────────────────────

async fn stats(State(st): State<AppState>) -> impl IntoResponse {
    let eng = st.engine.read().await;
    let sessions = st.memory.list_sessions().await;
    Json(json!({
        "engine": eng.engine.to_string(),
        "mode": eng.mode.to_string(),
        "vram_used_gb": null,
        "vram_total_gb": null,
        "active_sessions": sessions.len(),
        "history_len": 0,
        "model_loaded": true,
        "model_idle_seconds": 0.0,
    }))
}

async fn list_sessions(State(st): State<AppState>) -> impl IntoResponse {
    let sessions = st.memory.list_sessions().await;
    Json(json!({"sessions": sessions}))
}

async fn delete_session(State(st): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    st.memory.clear_session(&id).await;
    Json(json!({"message": format!("Sesja {id} usunięta")}))
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok", "version": "2.0.0"}))
}
