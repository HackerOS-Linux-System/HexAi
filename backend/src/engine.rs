use crate::{config::Config, error::{HexError, Result}};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{env, pin::Pin, sync::Arc};
use tokio_stream::StreamExt;

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

// ── Enums ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind { Local, #[default] Ollama, OpenAI, Candle }

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineKind::Local  => write!(f, "local"),
            EngineKind::Ollama => write!(f, "ollama"),
            EngineKind::OpenAI => write!(f, "openai"),
            EngineKind::Candle => write!(f, "candle"),
        }
    }
}

impl std::str::FromStr for EngineKind {
    type Err = ();
    fn from_str(s: &str) -> std::result::Result<Self, ()> {
        match s {
            "local"  => Ok(Self::Local),
            "ollama" => Ok(Self::Ollama),
            "openai" => Ok(Self::OpenAI),
            "candle" => Ok(Self::Candle),
            _        => Err(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Mode { #[default] General, Programista }

impl Mode {
    pub fn system_prompt(&self) -> &'static str {
        match self {
            Mode::General =>
                "Jesteś HexAi – inteligentnym, pomocnym asystentem AI. \
                 Odpowiadaj precyzyjnie, zwięźle i uprzejmie po polsku.",
            Mode::Programista =>
                "Jesteś HexAi – ekspertem programistycznym. \
                 Twój kod jest czysty, wydajny i zgodny z najlepszymi praktykami. \
                 Zawsze podajesz kompletne przykłady z obsługą błędów. \
                 Wyjaśniaj rozwiązania krok po kroku po polsku.",
        }
    }
    pub fn ollama_model(&self) -> &'static str {
        match self { Mode::General => "llama2", Mode::Programista => "codellama:7b-instruct" }
    }
    pub fn openai_model(&self) -> &'static str {
        match self { Mode::General => "gpt-4o-mini", Mode::Programista => "gpt-4o" }
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Mode::General => write!(f, "general"), Mode::Programista => write!(f, "programista") }
    }
}

// ── Message helpers ───────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmMessage { pub role: String, pub content: String }

pub fn build_messages(history: &[(String, String)], prompt: &str) -> Vec<LlmMessage> {
    let mut msgs = vec![];
    for (u, a) in history {
        msgs.push(LlmMessage { role: "user".into(),      content: u.clone() });
        msgs.push(LlmMessage { role: "assistant".into(), content: a.clone() });
    }
    msgs.push(LlmMessage { role: "user".into(), content: prompt.to_string() });
    msgs
}

/// Format history + prompt as a Llama-2/Mistral instruct string.
pub fn format_prompt(system: &str, history: &[(String, String)], prompt: &str) -> String {
    let mut out = format!("[INST] <<SYS>>\n{system}\n<</SYS>>\n\n");
    for (u, a) in history {
        out.push_str(&format!("{u} [/INST] {a} </s><s>[INST] "));
    }
    out.push_str(&format!("{prompt} [/INST]"));
    out
}

// ── Local engine (llama_cpp 0.2) ──────────────────────────────────

#[derive(Clone, Debug)]
pub struct LocalParams {
    pub n_ctx:        u32,
    pub n_threads:    u32,
    pub n_gpu_layers: u32,
    pub temp:         f32,
    pub top_p:        f32,
    pub max_tokens:   u32,
}

impl Default for LocalParams {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32).unwrap_or(4);
        Self {
            n_ctx:        env_u32("HEXAI_MODEL_CTX",        4096),
            n_threads:    env_u32("HEXAI_MODEL_THREADS",    cpus),
            n_gpu_layers: env_u32("HEXAI_MODEL_GPU_LAYERS", 0),
            temp:         env_f32("HEXAI_MODEL_TEMP",       0.7),
            top_p:        env_f32("HEXAI_MODEL_TOP_P",      0.9),
            max_tokens:   env_u32("HEXAI_MODEL_MAX_TOKENS", 1024),
        }
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[derive(Clone)]
pub struct LocalEngine {
    model:  Arc<llama_cpp::LlamaModel>,
    params: LocalParams,
}

impl LocalEngine {
    pub fn load(model_path: &str, params: LocalParams) -> Result<Self> {
        use llama_cpp::{LlamaModel, LlamaParams};

        if !std::path::Path::new(model_path).exists() {
            return Err(HexError::Engine(format!(
                "Model GGUF nie znaleziony: {model_path}\n\n\
                 Pobierz model GGUF (jednorazowo):\n\
                   # Mistral 7B (~4 GB RAM):\n\
                   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf\n\n\
                   # Llama 3.2 3B (~2 GB RAM, szybki):\n\
                   wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf\n\n\
                 Następnie ustaw:\n\
                   export HEXAI_ENGINE=local\n\
                   export HEXAI_MODEL_PATH=/ścieżka/do/model.gguf"
            )));
        }

        tracing::info!("Ładowanie modelu: {model_path}");
        tracing::info!("  ctx={} threads={} gpu_layers={}",
            params.n_ctx, params.n_threads, params.n_gpu_layers);

        let llama_params = LlamaParams {
            n_gpu_layers: params.n_gpu_layers,
            ..Default::default()
        };

        let model = LlamaModel::load_from_file(model_path, llama_params)
            .map_err(|e| HexError::Engine(format!("Błąd ładowania modelu: {e}")))?;

        tracing::info!("✓ Model załadowany lokalnie");
        Ok(Self { model: Arc::new(model), params })
    }

    pub async fn generate_stream(
        &self,
        system:  &str,
        history: &[(String, String)],
        prompt:  &str,
    ) -> Result<TokenStream> {
        use llama_cpp::standard_sampler::StandardSampler;
        use llama_cpp::SessionParams;

        let model       = Arc::clone(&self.model);
        let full_prompt = format_prompt(system, history, prompt);
        let params      = self.params.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String>>(512);

        tokio::task::spawn_blocking(move || {
            let session_params = SessionParams {
                n_ctx: params.n_ctx,
                ..Default::default()
            };

            // Clone model for TokensToStrings (Arc::clone gives us another Arc,
            // we deref to get LlamaModel – requires LlamaModel: Clone)
            // If LlamaModel isn't Clone, we pass (*model).clone() via Arc::try_unwrap
            // or keep a raw clone. Use Arc::as_ref() and rely on model impl.
            let model_ref = Arc::clone(&model);

            let mut session = match model_ref.create_session(session_params) {
                Ok(s)  => s,
                Err(e) => {
                    let _ = tx.blocking_send(Err(HexError::Engine(e.to_string())));
                    return;
                }
            };

            if let Err(e) = session.advance_context(full_prompt.as_str()) {
                let _ = tx.blocking_send(Err(HexError::Engine(e.to_string())));
                return;
            }

            let sampler = StandardSampler::default();
            let completions = match session.start_completing_with(sampler, params.max_tokens as usize) {
                Ok(c)  => c,
                Err(e) => { let _ = tx.blocking_send(Err(HexError::Engine(e.to_string()))); return; }
            };

            // TokensToStrings::new(completions, model) – model needed for decode
            use llama_cpp::TokensToStrings;
            // LlamaModel stored in Arc; need owned value – clone via deref if Clone,
            // otherwise use the model stored directly in Arc
            let model_owned = (*model_ref).clone();
            let token_strings = TokensToStrings::new(completions, model_owned);
            for token_str in token_strings {
                if tx.blocking_send(Ok(token_str)).is_err() { break; }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    pub async fn generate_sync(
        &self,
        system:  &str,
        history: &[(String, String)],
        prompt:  &str,
    ) -> Result<String> {
        let mut stream = self.generate_stream(system, history, prompt).await?;
        let mut buf = String::new();
        while let Some(tok) = stream.next().await { buf.push_str(&tok?); }
        Ok(buf.trim().to_string())
    }
}

// ── Ollama backend ────────────────────────────────────────────────

#[derive(Clone)]
struct OllamaBackend { http: reqwest::Client, base_url: String }

impl OllamaBackend {
    fn new(base_url: &str) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build().unwrap(),
            base_url: base_url.to_string(),
        }
    }

    async fn stream(&self, model: &str, system: &str, messages: Vec<LlmMessage>) -> Result<TokenStream> {
        #[derive(Serialize)]   struct Req<'a> { model: &'a str, messages: Vec<LlmMessage>, stream: bool }
        #[derive(Deserialize)] struct Chunk   { message: Msg, done: bool }
        #[derive(Deserialize)] struct Msg     { content: String }

        let mut all = vec![LlmMessage { role: "system".into(), content: system.to_string() }];
        all.extend(messages);

        let resp = self.http.post(format!("{}/api/chat", self.base_url))
            .json(&Req { model, messages: all, stream: true })
            .send().await
            .map_err(|e| HexError::Engine(format!("Ollama unreachable: {e}. Is `ollama serve` running?")))?;

        if !resp.status().is_success() {
            return Err(HexError::Engine(format!(
                "Ollama HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default()
            )));
        }

        let byte_stream = resp.bytes_stream();
        let s = async_stream::stream! {
            tokio::pin!(byte_stream);
            while let Some(item) = byte_stream.next().await {
                match item {
                    Err(e) => { yield Err(HexError::Engine(e.to_string())); break; }
                    Ok(bytes) => {
                        for raw in std::str::from_utf8(&bytes).unwrap_or("").lines() {
                            if raw.is_empty() { continue; }
                            if let Ok(chunk) = serde_json::from_str::<Chunk>(raw) {
                                yield Ok(chunk.message.content);
                                if chunk.done { return; }
                            }
                        }
                    }
                }
            }
        };
        Ok(Box::pin(s))
    }

    async fn sync_(&self, model: &str, system: &str, messages: Vec<LlmMessage>) -> Result<String> {
        let mut stream = self.stream(model, system, messages).await?;
        let mut buf = String::new();
        while let Some(tok) = stream.next().await { buf.push_str(&tok?); }
        Ok(buf)
    }
}

// ── OpenAI-compatible backend ─────────────────────────────────────

#[derive(Clone)]
struct OpenAIBackend { http: reqwest::Client, base_url: String, api_key: String }

impl OpenAIBackend {
    fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build().unwrap(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key:  api_key.to_string(),
        }
    }

    async fn stream(&self, model: &str, system: &str, messages: Vec<LlmMessage>) -> Result<TokenStream> {
        #[derive(Serialize)]
        struct Req<'a> { model: &'a str, messages: Vec<LlmMessage>, stream: bool }
        #[derive(Deserialize)] struct Chunk  { choices: Vec<Choice> }
        #[derive(Deserialize)] struct Choice { delta: Delta }
        #[derive(Deserialize)] struct Delta  { #[serde(default)] content: String }

        let mut all = vec![LlmMessage { role: "system".into(), content: system.to_string() }];
        all.extend(messages);

        let resp = self.http
            .post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&Req { model, messages: all, stream: true })
            .send().await
            .map_err(|e| HexError::Engine(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(HexError::Engine(format!(
                "OpenAI HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default()
            )));
        }

        let byte_stream = resp.bytes_stream();
        let s = async_stream::stream! {
            tokio::pin!(byte_stream);
            let mut buf = String::new();
            while let Some(item) = byte_stream.next().await {
                match item {
                    Err(e) => { yield Err(HexError::Engine(e.to_string())); break; }
                    Ok(bytes) => {
                        buf.push_str(std::str::from_utf8(&bytes).unwrap_or(""));
                        while let Some(nl) = buf.find('\n') {
                            let line = buf[..nl].trim().to_string();
                            buf = buf[nl+1..].to_string();
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" { return; }
                                if let Ok(chunk) = serde_json::from_str::<Chunk>(data) {
                                    for c in chunk.choices {
                                        if !c.delta.content.is_empty() {
                                            yield Ok(c.delta.content);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        Ok(Box::pin(s))
    }

    async fn sync_(&self, model: &str, system: &str, messages: Vec<LlmMessage>) -> Result<String> {
        let mut stream = self.stream(model, system, messages).await?;
        let mut buf = String::new();
        while let Some(tok) = stream.next().await { buf.push_str(&tok?); }
        Ok(buf)
    }
}

// ── Dispatcher ────────────────────────────────────────────────────

#[derive(Clone)]
enum Backend {
    Local(LocalEngine),
    Ollama(OllamaBackend),
    OpenAI(OpenAIBackend),
}

#[derive(Clone)]
pub struct LlmEngine {
    pub kind:       EngineKind,
    pub mode:       Mode,
    backend:        Backend,
    pub model_path: Option<String>,
}

impl LlmEngine {
    pub fn new(cfg: &Config) -> Self {
        let kind: EngineKind = env::var("HEXAI_ENGINE")
            .ok().and_then(|v| v.parse().ok()).unwrap_or_default();

        let (backend, model_path) = match &kind {
            EngineKind::Local => {
                let path = env::var("HEXAI_MODEL_PATH").unwrap_or_default();
                match LocalEngine::load(&path, LocalParams::default()) {
                    Ok(engine) => (Backend::Local(engine), Some(path)),
                    Err(e) => {
                        tracing::error!("{e}");
                        tracing::warn!("Fallback → Ollama");
                        (Backend::Ollama(OllamaBackend::new(&cfg.ollama_url)), None)
                    }
                }
            }
            EngineKind::OpenAI => {
                let key  = env::var("OPENAI_API_KEY").unwrap_or_default();
                let base = env::var("OPENAI_API_BASE")
                    .unwrap_or_else(|_| "https://api.openai.com".into());
                (Backend::OpenAI(OpenAIBackend::new(&base, &key)), None)
            }
            _ => (Backend::Ollama(OllamaBackend::new(&cfg.ollama_url)), None),
        };

        Self { kind, mode: Mode::default(), backend, model_path }
    }

    pub async fn generate_stream(
        &self, prompt: &str, history: &[(String, String)],
    ) -> Result<TokenStream> {
        let sys  = self.mode.system_prompt();
        let msgs = build_messages(history, prompt);
        match &self.backend {
            Backend::Local(e)  => e.generate_stream(sys, history, prompt).await,
            Backend::Ollama(e) => e.stream(self.mode.ollama_model(), sys, msgs).await,
            Backend::OpenAI(e) => e.stream(self.mode.openai_model(), sys, msgs).await,
        }
    }

    pub async fn generate_sync(
        &self, prompt: &str, history: &[(String, String)],
    ) -> Result<String> {
        let sys  = self.mode.system_prompt();
        let msgs = build_messages(history, prompt);
        match &self.backend {
            Backend::Local(e)  => e.generate_sync(sys, history, prompt).await,
            Backend::Ollama(e) => e.sync_(self.mode.ollama_model(), sys, msgs).await,
            Backend::OpenAI(e) => e.sync_(self.mode.openai_model(), sys, msgs).await,
        }
    }

    pub fn count_tokens(msgs: &[(String, String)]) -> usize {
        msgs.iter().map(|(u, a)| (u.len() + a.len()) / 4).sum()
    }

    pub fn trim_history(history: &mut Vec<(String, String)>, max_tokens: usize) {
        while Self::count_tokens(history) > max_tokens && history.len() > 1 {
            history.remove(0);
        }
    }

    pub fn info(&self) -> String {
        match &self.backend {
            Backend::Local(e) => format!(
                "local/llama_cpp | ctx={} threads={} gpu_layers={}",
                e.params.n_ctx, e.params.n_threads, e.params.n_gpu_layers
            ),
            Backend::Ollama(_) => format!("ollama → {}", self.mode.ollama_model()),
            Backend::OpenAI(_) => format!("openai → {}", self.mode.openai_model()),
        }
    }
}
