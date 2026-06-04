use hexai_backend::{
    config::Config,
    engine::LlmEngine,
    memory::PersistentMemory,
    router::build_router,
    state::AppState,
};
use std::net::SocketAddr;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env()
            .add_directive("hexai_backend=info".parse()?))
        .init();

    let cfg = Config::default();

    // Try Redis
    let redis_mgr = redis::Client::open(cfg.redis_url.clone())
        .ok()
        .and_then(|c| {
            tokio::runtime::Handle::current().block_on(async {
                redis::aio::ConnectionManager::new(c).await.ok()
            })
        })
        .map(std::sync::Arc::new);

    if redis_mgr.is_none() {
        tracing::warn!("Redis unavailable – using in-memory session store.");
    }

    let memory = PersistentMemory::new(redis_mgr, cfg.session_ttl_secs);
    let engine = LlmEngine::new(&cfg);
    let state  = AppState::new(cfg.clone(), engine, memory);
    let router = build_router(state);

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    info!("HexAi backend listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
