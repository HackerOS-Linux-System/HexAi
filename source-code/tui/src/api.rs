use serde::Deserialize;

pub const API_BASE: &str = "http://localhost:8000";

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Stats {
    pub engine: String,
    pub mode: String,
    pub vram_used_gb: Option<f64>,
    pub vram_total_gb: Option<f64>,
    pub active_sessions: u32,
    pub model_loaded: bool,
    pub model_idle_seconds: f64,
}

pub async fn fetch_stats() -> anyhow::Result<Stats> {
    let s = reqwest::get(format!("{API_BASE}/stats")).await?.json().await?;
    Ok(s)
}

pub async fn set_engine(engine: &str) -> anyhow::Result<()> {
    reqwest::Client::new()
        .post(format!("{API_BASE}/engine"))
        .json(&serde_json::json!({"engine": engine}))
        .send().await?;
    Ok(())
}

pub async fn set_mode(mode: &str) -> anyhow::Result<()> {
    reqwest::Client::new()
        .post(format!("{API_BASE}/mode"))
        .json(&serde_json::json!({"mode": mode}))
        .send().await?;
    Ok(())
}

pub async fn stream_chat(
    session_id: Option<&str>,
    message: &str,
    tx: tokio::sync::mpsc::Sender<StreamEvent>,
) {
    let payload = serde_json::json!({
        "message": message,
        "stream": true,
        "session_id": session_id,
    });

    let client = reqwest::Client::new();
    let resp = match client
        .post(format!("{API_BASE}/chat"))
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => { let _ = tx.send(StreamEvent::Error(e.to_string())).await; return; }
    };

    use futures::StreamExt;
    let mut byte_stream = resp.bytes_stream();
    while let Some(chunk) = byte_stream.next().await {
        match chunk {
            Ok(b) => {
                if let Ok(s) = std::str::from_utf8(&b) {
                    let _ = tx.send(StreamEvent::Token(s.to_string())).await;
                }
            }
            Err(e) => { let _ = tx.send(StreamEvent::Error(e.to_string())).await; return; }
        }
    }
    let _ = tx.send(StreamEvent::Done).await;
}

#[derive(Debug)]
pub enum StreamEvent {
    Token(String),
    Done,
    Error(String),
}
