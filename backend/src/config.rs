use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub redis_url: String,
    pub ollama_url: String,
    pub chroma_path: String,
    pub serper_api_key: Option<String>,
    pub model_idle_secs: u64,
    pub session_ttl_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: env::var("HEXAI_HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: env::var("HEXAI_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8000),
            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".into()),
            ollama_url: env::var("OLLAMA_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
            chroma_path: env::var("CHROMA_PATH")
                .unwrap_or_else(|_| "./chroma_db".into()),
            serper_api_key: env::var("SERPER_API_KEY").ok(),
            model_idle_secs: 600,
            session_ttl_secs: 86400,
        }
    }
}
