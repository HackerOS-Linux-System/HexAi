use crate::error::Result;
use dashmap::DashMap;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::warn;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Turn {
    pub user: String,
    pub assistant: String,
}

#[derive(Clone)]
pub struct PersistentMemory {
    redis: Option<Arc<redis::aio::ConnectionManager>>,
    fallback: Arc<DashMap<String, Vec<Turn>>>,
    ttl: u64,
}

impl PersistentMemory {
    pub fn new(redis: Option<Arc<redis::aio::ConnectionManager>>, ttl: u64) -> Self {
        Self { redis, fallback: Arc::new(DashMap::new()), ttl }
    }

    pub async fn get_history(&self, session_id: &str) -> Vec<Turn> {
        if let Some(r) = &self.redis {
            let key = format!("session:{session_id}");
            let res: redis::RedisResult<Vec<String>> =
                r.clone().lrange(&key, 0, -1).await;
            match res {
                Ok(items) => {
                    return items
                        .iter()
                        .filter_map(|s| serde_json::from_str(s).ok())
                        .collect();
                }
                Err(e) => warn!("Redis get_history: {e}"),
            }
        }
        self.fallback
            .get(session_id)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    pub async fn add_message(&self, session_id: &str, user: &str, assistant: &str) {
        let turn = Turn { user: user.to_string(), assistant: assistant.to_string() };
        // Fallback first
        let mut entry = self.fallback.entry(session_id.to_string()).or_default();
        entry.push(turn.clone());
        if entry.len() > 50 {
            let len = entry.len();
            entry.drain(0..len - 50);
        }
        drop(entry);

        if let Some(r) = &self.redis {
            let key = format!("session:{session_id}");
            let json = serde_json::to_string(&turn).unwrap_or_default();
            let _: redis::RedisResult<()> = r.clone().rpush(&key, &json).await;
            let _: redis::RedisResult<()> = r.clone().expire(&key, self.ttl as i64).await;
        }
    }

    pub async fn clear_session(&self, session_id: &str) {
        self.fallback.remove(session_id);
        if let Some(r) = &self.redis {
            let key = format!("session:{session_id}");
            let _: redis::RedisResult<()> = r.clone().del(&key).await;
        }
    }

    pub async fn list_sessions(&self) -> Vec<String> {
        let mut sessions: Vec<String> = self.fallback.iter().map(|e| e.key().clone()).collect();
        if let Some(r) = &self.redis {
            let keys: redis::RedisResult<Vec<String>> = r.clone().keys("session:*").await;
            if let Ok(keys) = keys {
                for k in keys {
                    let id = k.trim_start_matches("session:").to_string();
                    if !sessions.contains(&id) {
                        sessions.push(id);
                    }
                }
            }
        }
        sessions
    }
}
