use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},  // std RwLock – guard is Send
};
use tracing::{info, warn};
use uuid::Uuid;

// ── Chunk ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocChunk {
    pub id:        String,
    pub text:      String,
    pub source:    String,
    pub chunk_idx: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding: Vec<f32>,
}

// ── Chunker ───────────────────────────────────────────────────────

pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize, source: &str) -> Vec<DocChunk> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() { return vec![]; }
    let mut chunks = vec![];
    let mut start  = 0usize;
    let mut idx    = 0usize;
    loop {
        let end   = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(DocChunk {
            id: Uuid::new_v4().to_string(),
            text: chunk,
            source: source.to_string(),
            chunk_idx: idx,
            embedding: vec![],
        });
        if end == words.len() { break; }
        start = end.saturating_sub(overlap);
        idx  += 1;
    }
    chunks
}

// ── TF-IDF ────────────────────────────────────────────────────────

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_string())
        .collect()
}

fn tfidf_vec(text: &str, idf: &HashMap<String, f32>) -> HashMap<String, f32> {
    let tokens = tokenize(text);
    let n = tokens.len() as f32;
    if n == 0.0 { return HashMap::new(); }
    let mut tf: HashMap<String, f32> = HashMap::new();
    for t in &tokens { *tf.entry(t.clone()).or_default() += 1.0; }
    tf.into_iter()
        .filter_map(|(t, c)| idf.get(&t).map(|w| (t, (c / n) * w)))
        .collect()
}

fn cosine(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
    let dot: f32 = a.iter().filter_map(|(k, v)| b.get(k).map(|w| v * w)).sum();
    let na: f32  = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32  = b.values().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn dense_cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32  = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32  = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

// ── Disk persistence ──────────────────────────────────────────────

#[derive(Serialize, Deserialize, Default)]
struct IndexFile { chunks: Vec<DocChunk> }

fn index_path(db_path: &str) -> String {
    format!("{}.rag.json", db_path.trim_end_matches(".db"))
}

// ── RagStore ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RagStore {
    docs:        Arc<DashMap<String, DocChunk>>,
    // std::sync::RwLock guard is Send – safe to hold briefly (never across .await)
    idf:         Arc<RwLock<HashMap<String, f32>>>,
    ollama_url:  String,
    embed_model: String,
    index_file:  String,
}

// DashMap + std::RwLock are both Send+Sync
unsafe impl Send for RagStore {}
unsafe impl Sync for RagStore {}

impl RagStore {
    pub fn new(db_path: &str, ollama_url: &str) -> Self {
        let store = Self {
            docs:        Arc::new(DashMap::new()),
            idf:         Arc::new(RwLock::new(HashMap::new())),
            ollama_url:  ollama_url.to_string(),
            embed_model: std::env::var("HEXAI_EMBED_MODEL")
                             .unwrap_or_else(|_| "nomic-embed-text".to_string()),
            index_file:  index_path(db_path),
        };
        store.load_from_disk();
        store
    }

    // ── Disk ──────────────────────────────────────────────────────

    fn load_from_disk(&self) {
        if !Path::new(&self.index_file).exists() { return; }
        match std::fs::read_to_string(&self.index_file)
            .and_then(|s| serde_json::from_str::<IndexFile>(&s)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)))
        {
            Ok(idx) => {
                let n = idx.chunks.len();
                for chunk in idx.chunks { self.docs.insert(chunk.id.clone(), chunk); }
                self.rebuild_idf_sync();
                info!("RAG: loaded {n} chunks from {}", self.index_file);
            }
            Err(e) => warn!("RAG: could not load index: {e}"),
        }
    }

    fn save_to_disk(&self) {
        let chunks: Vec<DocChunk> = self.docs.iter().map(|e| e.value().clone()).collect();
        if let Ok(s) = serde_json::to_string(&IndexFile { chunks }) {
            let _ = std::fs::write(&self.index_file, s);
        }
    }

    // ── IDF ───────────────────────────────────────────────────────

    fn rebuild_idf_sync(&self) {
        let n = self.docs.len() as f32;
        if n == 0.0 { return; }
        let mut df: HashMap<String, f32> = HashMap::new();
        for entry in self.docs.iter() {
            let unique: std::collections::HashSet<String> =
                tokenize(&entry.text).into_iter().collect();
            for t in unique { *df.entry(t).or_default() += 1.0; }
        }
        let idf: HashMap<String, f32> = df.into_iter()
            .map(|(t, d)| (t, (1.0 + n / (1.0 + d)).ln()))
            .collect();
        *self.idf.write().unwrap() = idf;
    }

    // ── Embeddings ────────────────────────────────────────────────

    async fn embed(&self, text: &str) -> Option<Vec<f32>> {
        #[derive(serde::Serialize)]   struct Req<'a> { model: &'a str, prompt: &'a str }
        #[derive(serde::Deserialize)] struct Resp    { embedding: Vec<f32> }
        reqwest::Client::new()
            .post(format!("{}/api/embeddings", self.ollama_url))
            .json(&Req { model: &self.embed_model, prompt: text })
            .send().await.ok()?
            .json::<Resp>().await.ok()
            .map(|r| r.embedding)
    }

    // ── Add ───────────────────────────────────────────────────────

    pub async fn add_documents(&self, mut chunks: Vec<DocChunk>) {
        for chunk in &mut chunks {
            if let Some(emb) = self.embed(&chunk.text).await {
                chunk.embedding = emb;
            }
        }
        for chunk in chunks { self.docs.insert(chunk.id.clone(), chunk); }
        self.rebuild_idf_sync();
        self.save_to_disk();
    }

    pub fn count(&self) -> usize { self.docs.len() }

    // ── Search ────────────────────────────────────────────────────
    // KEY RULE: never hold a lock guard across an .await point.
    // We collect what we need under the lock, drop the guard, then await.

    pub async fn search(&self, query: &str, k: usize) -> Vec<String> {
        if self.docs.is_empty() { return vec![]; }

        // 1. Collect IDF snapshot (sync, no await) – guard dropped at end of block
        let idf_snapshot: HashMap<String, f32> = {
            self.idf.read().unwrap().clone()  // clone so guard drops immediately
        };

        // 2. Compute query sparse vector (sync)
        let q_sparse = tfidf_vec(query, &idf_snapshot);

        // 3. Collect all doc data (sync, DashMap iteration is Send-safe)
        let doc_data: Vec<(String, Vec<f32>, HashMap<String, f32>)> = self.docs.iter()
            .map(|e| {
                let chunk = e.value();
                let sparse = tfidf_vec(&chunk.text, &idf_snapshot);
                (chunk.text.clone(), chunk.embedding.clone(), sparse)
            })
            .collect();

        // 4. Dense query embedding (async – no locks held here)
        let q_dense = self.embed(query).await;

        // 5. Score all docs (sync)
        let mut scored: Vec<(f32, String)> = doc_data.into_iter()
            .map(|(text, emb, sparse)| {
                let s = cosine(&q_sparse, &sparse);
                let d = q_dense.as_ref()
                    .filter(|_| !emb.is_empty())
                    .map(|qd| dense_cosine(qd, &emb))
                    .unwrap_or(0.0);
                let score = if d > 0.0 { 0.4 * s + 0.6 * d } else { s };
                (score, text)
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(k).map(|(_, t)| t).collect()
    }

    pub fn search_sync(&self, query: &str, k: usize) -> Vec<String> {
        if self.docs.is_empty() { return vec![]; }
        let idf = self.idf.read().unwrap().clone();
        let q   = tfidf_vec(query, &idf);
        let mut scored: Vec<(f32, String)> = self.docs.iter()
            .map(|e| (cosine(&q, &tfidf_vec(&e.text, &idf)), e.text.clone()))
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(k).map(|(_, t)| t).collect()
    }
}
