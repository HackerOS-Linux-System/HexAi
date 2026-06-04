use crate::{
    config::Config,
    engine::{Engine, LlmEngine, Mode},
    memory::PersistentMemory,
    profiler::UserProfiler,
    rag::RagStore,
};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    pub engine: Arc<RwLock<LlmEngine>>,
    pub memory: Arc<PersistentMemory>,
    pub rag: Arc<RagStore>,
    pub profiler: Arc<UserProfiler>,
}

impl AppState {
    pub fn new(
        cfg: Config,
        engine: LlmEngine,
        memory: PersistentMemory,
    ) -> Self {
        Self {
            cfg: Arc::new(cfg),
            engine: Arc::new(RwLock::new(engine)),
            memory: Arc::new(memory),
            rag: Arc::new(RagStore::new()),
            profiler: Arc::new(UserProfiler::new()),
        }
    }

    pub async fn set_engine_kind(&self, kind: Engine) {
        self.engine.write().await.engine = kind;
    }

    pub async fn set_mode(&self, mode: Mode) {
        self.engine.write().await.mode = mode;
    }
}
