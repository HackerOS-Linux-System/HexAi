use crate::{config::Config, error::{HexError, Result}};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{env, pin::Pin};
use tokio_stream::StreamExt;

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

// ── Engine / Mode enums ───────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind { #[default] Ollama, OpenAI, Candle }

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineKind::Ollama => write!(f, "ollama"),
            EngineKind::OpenAI => write!(f, "openai"),
            EngineKind::Candle => write!(f, "candle"),
        }
    }
}

impl std::str::FromStr for EngineKind {
    type Err = ();
    fn from_str(s: &str) -> std::result::Result<Self, ()> {
        match s { "ollama" => Ok(Self::Ollama), "openai" => Ok(Self::OpenAI), "candle" => Ok(Self::Candle), _ => Err(()) }
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
                 Twój kod jest czysty, wydajny, dobrze skomentowany i zgodny z najlepszymi praktykami. \
                 Zawsze podajesz kompletne przykłady z obsługą błędów. \
                 Wyjaśniaj swoje rozwiązania krok po kroku po polsku.",
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

// ── Message ───────────────────────────────────────────────────────

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

// ── Ollama backend ────────────────────────────────────────────────

#[derive(Clone)]
struct OllamaBackend { http: reqwest::Client, base_url: String }

impl OllamaBackend {
    fn new(base_url: &str) -> Self {
        Self { http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(120)).build().unwrap(), base_url: base_url.to_string() }
    }

    async fn stream(&self, model: &str, system: &str, messages: Vec<LlmMessage>) -> Result<TokenStream> {
        #[derive(Serialize)]   struct Req<'a> { model: &'a str, messages: Vec<LlmMessage>, stream: bool }
        #[derive(Deserialize)] struct Chunk   { message: Msg, done: bool }
        #[derive(Deserialize)] struct Msg     { content: String }

        let mut all = vec![LlmMessage { role: "system".into(), content: system.to_string() }];
        all.extend(messages);

        let resp = self.http.post(format!("{}/api/chat", self.base_url))
            .json(&Req { model, messages: all, stream: true })
            .send().await.map_err(|e| HexError::Engine(format!("Ollama unreachable: {e}. Is `ollama serve` running?")))?;

        if !resp.status().is_success() {
            return Err(HexError::Engine(format!("Ollama HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default())));
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

    async fn sync(&self, model: &str, system: &str, messages: Vec<LlmMessage>) -> Result<String> {
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
            http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(120)).build().unwrap(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
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

        let resp = self.http.post(format!("{}/v1/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&Req { model, messages: all, stream: true })
            .send().await.map_err(|e| HexError::Engine(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(HexError::Engine(format!("OpenAI HTTP {}: {}", resp.status(), resp.text().await.unwrap_or_default())));
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
                                        if !c.delta.content.is_empty() { yield Ok(c.delta.content); }
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

    async fn sync(&self, model: &str, system: &str, messages: Vec<LlmMessage>) -> Result<String> {
        let mut stream = self.stream(model, system, messages).await?;
        let mut buf = String::new();
        while let Some(tok) = stream.next().await { buf.push_str(&tok?); }
        Ok(buf)
    }
}

// ── Dispatcher ────────────────────────────────────────────────────

#[derive(Clone)]
enum Backend { Ollama(OllamaBackend), OpenAI(OpenAIBackend) }

#[derive(Clone)]
pub struct LlmEngine {
    pub kind: EngineKind,
    pub mode: Mode,
    backend: Backend,
}

impl LlmEngine {
    pub fn new(cfg: &Config) -> Self {
        let kind: EngineKind = env::var("HEXAI_ENGINE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_default();

        let backend = match &kind {
            EngineKind::OpenAI => {
                let key  = env::var("OPENAI_API_KEY").unwrap_or_default();
                let base = env::var("OPENAI_API_BASE").unwrap_or_else(|_| "https://api.openai.com".into());
                Backend::OpenAI(OpenAIBackend::new(&base, &key))
            }
            _ => Backend::Ollama(OllamaBackend::new(&cfg.ollama_url)),
        };

        Self { kind, mode: Mode::default(), backend }
    }

    pub async fn generate_stream(&self, prompt: &str, history: &[(String, String)]) -> Result<TokenStream> {
        let msgs   = build_messages(history, prompt);
        let system = self.mode.system_prompt();
        match &self.backend {
            Backend::Ollama(b)  => b.stream(self.mode.ollama_model(), system, msgs).await,
            Backend::OpenAI(b)  => b.stream(self.mode.openai_model(), system, msgs).await,
        }
    }

    pub async fn generate_sync(&self, prompt: &str, history: &[(String, String)]) -> Result<String> {
        let msgs   = build_messages(history, prompt);
        let system = self.mode.system_prompt();
        match &self.backend {
            Backend::Ollama(b)  => b.sync(self.mode.ollama_model(), system, msgs).await,
            Backend::OpenAI(b)  => b.sync(self.mode.openai_model(), system, msgs).await,
        }
    }

    /// Rough token count (≈ chars/4) for history trimming
    pub fn count_tokens(msgs: &[(String, String)]) -> usize {
        msgs.iter().map(|(u, a)| (u.len() + a.len()) / 4).sum()
    }

    /// Trim history so total tokens stays below limit
    pub fn trim_history(history: &mut Vec<(String, String)>, max_tokens: usize) {
        while Self::count_tokens(history) > max_tokens && history.len() > 1 {
            history.remove(0);
        }
    }
}
