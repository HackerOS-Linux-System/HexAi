use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use chrono::{Duration, Utc};
use dashmap::DashMap;
use governor::{
    clock::DefaultClock, state::keyed::DefaultKeyedStateStore, Quota, RateLimiter,
};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::{net::IpAddr, num::NonZeroU32, sync::Arc};

use crate::config::Config;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    pub sub: String,
    pub exp: i64,
    pub adm: bool,
}

pub fn mint_token(username: &str, is_admin: bool, secret: &str, hours: i64) -> anyhow::Result<String> {
    let claims = Claims {
        sub: username.to_string(),
        exp: (Utc::now() + Duration::hours(hours)).timestamp(),
        adm: is_admin,
    };
    Ok(encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_bytes()))?)
}

pub fn verify_token(token: &str, secret: &str) -> Option<Claims> {
    decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::new(Algorithm::HS256),
    ).ok().map(|d| d.claims)
}

type Limiter = RateLimiter<IpAddr, DefaultKeyedStateStore<IpAddr>, DefaultClock>;

pub fn build_limiter(rpm: u32) -> Arc<Limiter> {
    let quota = Quota::per_minute(NonZeroU32::new(rpm.max(1)).unwrap());
    Arc::new(RateLimiter::keyed(quota))
}

#[derive(Clone)]
pub struct AuthState {
    pub limiter:      Arc<Limiter>,
    pub jwt_secret:   String,
    pub auth_enabled: bool,
}

impl AuthState {
    pub fn new(cfg: &Config) -> Self {
        Self {
            limiter:      build_limiter(cfg.rate_limit_rpm),
            jwt_secret:   cfg.jwt_secret.clone(),
            auth_enabled: cfg.auth_enabled,
        }
    }
}

pub async fn auth_middleware(
    State(auth): State<Arc<AuthState>>,
    headers: HeaderMap,
    mut req: Request,
    next: Next,
) -> Response {
    let ip: IpAddr = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));

    if auth.limiter.check_key(&ip).is_err() {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(serde_json::json!({ "error": "Rate limit exceeded. Try again in a minute." })),
        ).into_response();
    }

    if auth.auth_enabled {
        let token = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .unwrap_or("");

        match verify_token(token, &auth.jwt_secret) {
            Some(claims) => { req.extensions_mut().insert(claims); }
            None => return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "Invalid or missing token." })),
            ).into_response(),
        }
    }

    next.run(req).await
}

#[derive(Deserialize)]
pub struct LoginReq { pub username: String, pub password: String }

static USERS: Lazy<DashMap<String, (String, bool)>> = Lazy::new(|| {
    let m = DashMap::new();
    let pass = std::env::var("HEXAI_ADMIN_PASS").unwrap_or_else(|_| "hexai-admin".to_string());
    m.insert("admin".to_string(), (pass, true));
    m
});

pub async fn login(
    State(auth): State<Arc<AuthState>>,
    Json(req): Json<LoginReq>,
) -> impl IntoResponse {
    match USERS.get(&req.username) {
        Some(entry) if entry.0 == req.password => {
            match mint_token(&req.username, entry.1, &auth.jwt_secret, 24) {
                Ok(token) => (StatusCode::OK, Json(serde_json::json!({ "token": token }))).into_response(),
                Err(e)    => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
            }
        }
        _ => (StatusCode::UNAUTHORIZED, Json(serde_json::json!({ "error": "Invalid credentials." }))).into_response(),
    }
}
