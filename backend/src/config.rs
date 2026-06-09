use std::env;

#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub redis_url: String,
    pub ollama_url: String,
    pub serper_api_key: Option<String>,
    pub session_ttl_secs: u64,
    pub session_max_tokens: usize,
    pub db_path: String,
    // Auth
    pub jwt_secret: String,
    pub auth_enabled: bool,
    // Rate limiting (requests per minute per IP)
    pub rate_limit_rpm: u32,
    // CORS – comma-separated allowed origins, "*" = any
    pub cors_origins: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        let cors_raw = env::var("HEXAI_CORS_ORIGINS").unwrap_or_else(|_| "*".into());
        let cors_origins = cors_raw.split(',').map(|s| s.trim().to_string()).collect();
        Self {
            host:               env::var("HEXAI_HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port:               env::var("HEXAI_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8000),
            redis_url:          env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".into()),
            ollama_url:         env::var("OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
            serper_api_key:     env::var("SERPER_API_KEY").ok(),
            session_ttl_secs:   86400,
            session_max_tokens: 4096,
            db_path:            env::var("HEXAI_DB_PATH").unwrap_or_else(|_| "./hexai.db".into()),
            jwt_secret:         env::var("HEXAI_JWT_SECRET").unwrap_or_else(|_| "change-me-in-production".into()),
            auth_enabled:       env::var("HEXAI_AUTH").map(|v| v == "1" || v == "true").unwrap_or(false),
            rate_limit_rpm:     env::var("HEXAI_RATE_LIMIT_RPM").ok().and_then(|v| v.parse().ok()).unwrap_or(60),
            cors_origins,
        }
    }
}
