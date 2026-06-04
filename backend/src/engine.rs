use crate::{config::Config, error::{HexError, Result}};
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{pin::Pin, sync::Arc};
use tokio_stream::StreamExt;

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine { Transformers, Ollama }

impl std::fmt::Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Engine::Transformers => write!(f, "transformers"), Engine::Ollama => write!(f, "ollama") }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode { General, Programista }

impl Mode {
    pub fn system_prompt(&self) -> &'static str {
        match self {
            Mode::General => "Jesteś HexAi – inteligentnym, pomocnym asystentem AI. Odpowiadaj precyzyjnie, zwięźle i uprzejmie.",
            Mode::Programista => "Jesteś HexAi – ekspertem programistycznym. Twój kod jest czysty, wydajny, dobrze skomentowany i zgodny z najlepszymi praktykami. Zawsze podajesz kompletne przykłady z obsługą błędów. Wyjaśniaj swoje rozwiązania krok po kroku.",
        }
    }

    pub fn ollama_model(&self) -> &'static str {
        match self { Mode::General => "llama2", Mode::Programista => "codellama:7b-instruct" }
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Mode::General => write!(f, "general"), Mode::Programista => write!(f, "programista") }
    }
}

// ── Message history for LLM ──────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmMessage { pub role: String, pub content: String }

// ── Ollama client ────────────────────────────────────────────────

#[derive(Clone)]
pub struct OllamaClient {
    http: reqwest::Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: String) -> Self {
        Self { http: reqwest::Client::new(), base_url }
    }

    pub async fn chat_stream(
        &self,
        model: &str,
        system: &str,
        messages: Vec<LlmMessage>,
    ) -> Result<TokenStream> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            messages: Vec<LlmMessage>,
            stream: bool,
        }
        #[derive(Deserialize)]
        struct Chunk { message: MsgContent, done: bool }
        #[derive(Deserialize)]
        struct MsgContent { content: String }

        let mut all = vec![LlmMessage { role: "system".into(), content: system.to_string() }];
        all.extend(messages);

        let resp = self.http
            .post(format!("{}/api/chat", self.base_url))
            .json(&Req { model, messages: all, stream: true })
            .send()
            .await
            .map_err(|e| HexError::Engine(e.to_string()))?;

        let byte_stream = resp.bytes_stream();
        let s = async_stream::stream! {
            tokio::pin!(byte_stream);
            while let Some(item) = byte_stream.next().await {
                match item {
                    Err(e) => { yield Err(HexError::Engine(e.to_string())); break; }
                    Ok(bytes) => {
                        if let Ok(line) = std::str::from_utf8(&bytes) {
                            for raw in line.lines() {
                                if raw.is_empty() { continue; }
                                if let Ok(chunk) = serde_json::from_str::<Chunk>(raw) {
                                    yield Ok(chunk.message.content);
                                    if chunk.done { return; }
                                }
                            }
                        }
                    }
                }
            }
        };
        Ok(Box::pin(s))
    }

    pub async fn chat_sync(
        &self,
        model: &str,
        system: &str,
        messages: Vec<LlmMessage>,
    ) -> Result<String> {
        let mut stream = self.chat_stream(model, system, messages).await?;
        let mut buf = String::new();
        while let Some(token) = stream.next().await {
            buf.push_str(&token?);
        }
        Ok(buf)
    }
}

// ── Engine dispatcher ────────────────────────────────────────────

#[derive(Clone)]
pub struct LlmEngine {
    pub engine: Engine,
    pub mode: Mode,
    ollama: OllamaClient,
}

impl LlmEngine {
    pub fn new(cfg: &Config) -> Self {
        Self {
            engine: Engine::Transformers,
            mode: Mode::General,
            ollama: OllamaClient::new(cfg.ollama_url.clone()),
        }
    }

    pub fn build_messages(
        history: &[(String, String)],
        prompt: &str,
    ) -> Vec<LlmMessage> {
        let mut msgs = vec![];
        for (u, a) in history {
            msgs.push(LlmMessage { role: "user".into(), content: u.clone() });
            msgs.push(LlmMessage { role: "assistant".into(), content: a.clone() });
        }
        msgs.push(LlmMessage { role: "user".into(), content: prompt.to_string() });
        msgs
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        history: &[(String, String)],
    ) -> Result<TokenStream> {
        let msgs = Self::build_messages(history, prompt);
        let system = self.mode.system_prompt();
        let model = self.mode.ollama_model();
        // Both engines currently delegate to Ollama.
        // For `transformers` engine you can spawn a local Python subprocess
        // or a candle-based model here.
        self.ollama.chat_stream(model, system, msgs).await
    }

    pub async fn generate_sync(
        &self,
        prompt: &str,
        history: &[(String, String)],
    ) -> Result<String> {
        let msgs = Self::build_messages(history, prompt);
        let system = self.mode.system_prompt();
        let model = self.mode.ollama_model();
        self.ollama.chat_sync(model, system, msgs).await
    }
}
