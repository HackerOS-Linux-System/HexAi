use crate::{
    auth::AuthState,
    config::Config,
    engine::{EngineKind, LlmEngine, Mode},
    memory::PersistentMemory,
    profiler::UserProfiler,
    rag::RagStore,
};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub cfg:     Arc<Config>,
    pub engine:  Arc<RwLock<LlmEngine>>,
    pub memory:  Arc<PersistentMemory>,
    pub rag:     Arc<RagStore>,
    pub profiler: Arc<UserProfiler>,
    pub auth:    Arc<AuthState>,
}

impl AppState {
    pub fn new(cfg: Config, engine: LlmEngine, memory: PersistentMemory) -> Self {
        let rag     = RagStore::new(&cfg.db_path, &cfg.ollama_url);
        let auth    = AuthState::new(&cfg);
        Self {
            rag:     Arc::new(rag),
            auth:    Arc::new(auth),
            engine:  Arc::new(RwLock::new(engine)),
            memory:  Arc::new(memory),
            profiler: Arc::new(UserProfiler::new()),
            cfg:     Arc::new(cfg),
        }
    }

    pub async fn set_engine_kind(&self, kind: EngineKind) {
        self.engine.write().await.kind = kind;
    }
    pub async fn set_mode(&self, mode: Mode) {
        self.engine.write().await.mode = mode;
    }
}
