use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HexError {
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Bad request: {0}")]
    BadRequest(String),
    #[error("Engine error: {0}")]
    Engine(String),
    #[error("Internal: {0}")]
    Internal(String),
}

impl IntoResponse for HexError {
    fn into_response(self) -> Response {
        let (status, msg) = match &self {
            HexError::NotFound(_)   => (StatusCode::NOT_FOUND, self.to_string()),
            HexError::BadRequest(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            _                       => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        (status, Json(json!({ "error": msg }))).into_response()
    }
}

pub type Result<T> = std::result::Result<T, HexError>;
