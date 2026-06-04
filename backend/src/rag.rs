use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocChunk {
    pub id: String,
    pub text: String,
    pub source: String,
    pub chunk_idx: usize,
}

#[derive(Clone, Default)]
pub struct RagStore {
    docs: Arc<DashMap<String, DocChunk>>,
}

impl RagStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_documents(&self, chunks: Vec<DocChunk>) {
        for chunk in chunks {
            self.docs.insert(chunk.id.clone(), chunk);
        }
    }

    /// Simple TF-IDF inspired keyword search
    pub fn search(&self, query: &str, k: usize) -> Vec<String> {
        let query_words: Vec<&str> = query.split_whitespace().collect();
        if query_words.is_empty() || self.docs.is_empty() {
            return vec![];
        }

        let mut scored: Vec<(f64, String)> = self
            .docs
            .iter()
            .map(|entry| {
                let text_lower = entry.text.to_lowercase();
                let score: f64 = query_words
                    .iter()
                    .map(|w| {
                        let w_lower = w.to_lowercase();
                        text_lower.matches(&w_lower).count() as f64
                    })
                    .sum();
                (score, entry.text.clone())
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(k).map(|(_, t)| t).collect()
    }

    pub fn count(&self) -> usize {
        self.docs.len()
    }
}

/// Split text into overlapping chunks
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize, source: &str) -> Vec<DocChunk> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = vec![];
    let mut start = 0usize;
    let mut idx = 0usize;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk_text = words[start..end].join(" ");
        chunks.push(DocChunk {
            id: Uuid::new_v4().to_string(),
            text: chunk_text,
            source: source.to_string(),
            chunk_idx: idx,
        });
        if end == words.len() {
            break;
        }
        start = if chunk_size > overlap { end - overlap } else { end };
        idx += 1;
    }
    chunks
}
