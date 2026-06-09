use crate::engine::LlmEngine;
use dashmap::DashMap;
use redis::AsyncCommands;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{info, warn};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Turn {
    pub user:      String,
    pub assistant: String,
}

fn open_db(path: &str) -> rusqlite::Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("
        PRAGMA journal_mode=WAL;
        PRAGMA busy_timeout=5000;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT NOT NULL,
            user_msg   TEXT NOT NULL,
            asst_msg   TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
        );
        CREATE INDEX IF NOT EXISTS idx_sessions
            ON sessions(session_id, created_at);
    ")?;
    Ok(conn)
}

#[derive(Clone)]
pub struct PersistentMemory {
    db:       Arc<StdMutex<Connection>>,
    redis:    Option<Arc<AsyncMutex<redis::aio::ConnectionManager>>>,
    fallback: Arc<DashMap<String, Vec<Turn>>>,
    ttl:      u64,
    max_tok:  usize,
}

impl PersistentMemory {
    pub fn new(
        db_path: &str,
        redis:   Option<Arc<AsyncMutex<redis::aio::ConnectionManager>>>,
        ttl:     u64,
        max_tokens: usize,
    ) -> Self {
        let conn = open_db(db_path).unwrap_or_else(|e| {
            warn!("SQLite {db_path}: {e} — używam :memory:");
            open_db(":memory:").expect("in-memory SQLite failed")
        });
        info!("SQLite session store: {db_path}");
        Self {
            db:       Arc::new(StdMutex::new(conn)),
            redis,
            fallback: Arc::new(DashMap::new()),
            ttl,
            max_tok: max_tokens,
        }
    }

    // ── helpers ───────────────────────────────────────────────────

    fn db_get(&self, session_id: &str) -> Vec<Turn> {
        let db = self.db.lock().unwrap();
        let mut stmt = match db.prepare_cached(
            "SELECT user_msg, asst_msg FROM sessions \
             WHERE session_id=? ORDER BY created_at ASC",
        ) {
            Ok(s)  => s,
            Err(_) => return vec![],
        };
        let mut rows = match stmt.query(params![session_id]) {
            Ok(r)  => r,
            Err(_) => return vec![],
        };
        let mut turns = vec![];
        while let Ok(Some(row)) = rows.next() {
            if let (Ok(u), Ok(a)) = (row.get::<_, String>(0), row.get::<_, String>(1)) {
                turns.push(Turn { user: u, assistant: a });
            }
        }
        turns
    }

    fn db_insert(&self, session_id: &str, user: &str, asst: &str) {
        let db = self.db.lock().unwrap();
        let _ = db.execute(
            "INSERT INTO sessions(session_id,user_msg,asst_msg) VALUES(?,?,?)",
            params![session_id, user, asst],
        );
    }

    fn db_delete(&self, session_id: &str) {
        let db = self.db.lock().unwrap();
        let _ = db.execute(
            "DELETE FROM sessions WHERE session_id=?",
            params![session_id],
        );
    }

    fn db_list(&self) -> Vec<String> {
        let db = self.db.lock().unwrap();
        let mut stmt = match db.prepare_cached(
            "SELECT DISTINCT session_id FROM sessions"
        ) {
            Ok(s)  => s,
            Err(_) => return vec![],
        };
        let mut rows = match stmt.query([]) {
            Ok(r)  => r,
            Err(_) => return vec![],
        };
        let mut ids = vec![];
        while let Ok(Some(row)) = rows.next() {
            if let Ok(id) = row.get::<_, String>(0) { ids.push(id); }
        }
        ids
    }

    // ── public API ────────────────────────────────────────────────

    pub async fn get_history(&self, session_id: &str) -> Vec<Turn> {
        // 1. Redis cache
        if let Some(r) = &self.redis {
            let key = format!("session:{session_id}");
            let mut conn = r.lock().await;
            if let Ok(items) = conn.lrange::<_, Vec<String>>(&key, 0, -1).await {
                let turns: Vec<Turn> = items.iter()
                    .filter_map(|s| serde_json::from_str(s).ok())
                    .collect();
                if !turns.is_empty() { return turns; }
            }
        }
        // 2. SQLite
        let turns = self.db_get(session_id);
        if !turns.is_empty() { return turns; }
        // 3. RAM fallback
        self.fallback.get(session_id).map(|v| v.clone()).unwrap_or_default()
    }

    pub async fn add_message(&self, session_id: &str, user: &str, assistant: &str) {
        let turn = Turn { user: user.to_string(), assistant: assistant.to_string() };

        // RAM
        {
            let mut entry = self.fallback.entry(session_id.to_string()).or_default();
            entry.push(turn.clone());
            if entry.len() > 100 { entry.drain(0..50); }
        }
        // SQLite
        self.db_insert(session_id, user, assistant);

        // Redis
        if let Some(r) = &self.redis {
            let key  = format!("session:{session_id}");
            let json = serde_json::to_string(&turn).unwrap_or_default();
            let mut conn = r.lock().await;
            let _: redis::RedisResult<()> = conn.rpush(&key, &json).await;
            let _: redis::RedisResult<()> = conn.expire(&key, self.ttl as i64).await;
        }
    }

    pub async fn get_trimmed_history(&self, session_id: &str) -> Vec<(String, String)> {
        let turns = self.get_history(session_id).await;
        let mut pairs: Vec<(String, String)> = turns.into_iter()
            .map(|t| (t.user, t.assistant))
            .collect();
        LlmEngine::trim_history(&mut pairs, self.max_tok);
        pairs
    }

    pub async fn clear_session(&self, session_id: &str) {
        self.fallback.remove(session_id);
        self.db_delete(session_id);
        if let Some(r) = &self.redis {
            let mut conn = r.lock().await;
            let _: redis::RedisResult<()> =
                conn.del(format!("session:{session_id}")).await;
        }
    }

    pub async fn list_sessions(&self) -> Vec<String> {
        let mut sessions = self.db_list();
        for e in self.fallback.iter() {
            if !sessions.contains(e.key()) {
                sessions.push(e.key().clone());
            }
        }
        sessions
    }
}
