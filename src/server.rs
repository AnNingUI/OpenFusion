//! HTTP server: axum router with four protocol endpoints.

use crate::config::Config;
use crate::engine::FusionEngine;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    middleware,
    response::Json,
    routing::{get, post},
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tower::limit::ConcurrencyLimitLayer;

mod health;
pub use health::{health_check, metrics};

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<FusionEngine>,
    pub config: Arc<Config>,
    pub start_time: Instant,
    pub request_count: Arc<AtomicU64>,
}

/// Build the full axum Router.
pub fn build_router(state: AppState) -> Router {
    let max_concurrent = state.config.server.max_concurrent_requests;

    Router::new()
        // OpenAI Chat Completions
        .route("/v1/chat/completions", post(handle_chat_completions))
        // OpenAI Responses
        .route("/v1/responses", post(handle_responses))
        // Google Generative AI (catch-all to handle `:generateContent` suffix)
        .route("/v1/google/{*path}", post(handle_google_genai))
        // Anthropic Messages
        .route("/v1/messages", post(handle_anthropic_messages))
        // Health & metrics
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            count_request,
        ))
        .layer(
            tower_http::cors::CorsLayer::permissive()
        )
        .layer(
            tower_http::trace::TraceLayer::new_for_http()
        )
        .layer(ConcurrencyLimitLayer::new(max_concurrent))
        .with_state(state)
}

/// Middleware to count incoming requests.
async fn count_request(
    State(state): State<AppState>,
    req: axum::http::Request<axum::body::Body>,
    next: middleware::Next,
) -> axum::response::Response {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    next.run(req).await
}

// ── Handlers ──

async fn handle_chat_completions(
    State(state): State<AppState>,
    Json(body): Json<crate::protocol::openai_completions::ChatCompletionRequest>,
) -> Result<Json<crate::protocol::openai_completions::ChatCompletionResponse>, (StatusCode, String)> {
    tracing::debug!("Chat completion requested for model: {}", body.model);
    let model_name = state.config.fusion.name.clone();
    let request = crate::protocol::openai_completions::to_intermediate(body)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let result = state.engine.execute(&request).await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    let content = result.synthesis.unwrap_or_default();
    let ir = crate::protocol::IntermediateResponse {
        content,
        finish_reason: "stop".into(),
        usage: Default::default(),
        cost_usd: result.total_cost_usd,
        duration_ms: result.duration_ms,
        model: model_name.clone(),
    };

    Ok(Json(crate::protocol::openai_completions::from_intermediate(ir, &model_name)))
}

async fn handle_responses(
    State(state): State<AppState>,
    Json(body): Json<crate::protocol::openai_responses::ResponsesRequest>,
) -> Result<Json<crate::protocol::openai_responses::ResponsesOut>, (StatusCode, String)> {
    tracing::debug!("Responses API requested for model: {}", body.model);
    let model_name = state.config.fusion.name.clone();
    let request = crate::protocol::openai_responses::to_intermediate(body)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let result = state.engine.execute(&request).await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    let content = result.synthesis.unwrap_or_default();
    let ir = crate::protocol::IntermediateResponse {
        content,
        finish_reason: "stop".into(),
        usage: Default::default(),
        cost_usd: result.total_cost_usd,
        duration_ms: result.duration_ms,
        model: model_name.clone(),
    };

    Ok(Json(crate::protocol::openai_responses::from_intermediate(ir, &model_name)))
}

async fn handle_google_genai(
    State(state): State<AppState>,
    axum::extract::Path(path): axum::extract::Path<String>,
    Json(body): Json<crate::protocol::google_genai::GenAiRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Parse model name from path like "models/gemini-2.5-flash:generateContent"
    let model = path
        .strip_prefix("models/")
        .and_then(|s| s.strip_suffix(":generateContent"))
        .or_else(|| {
            path.strip_prefix("models/")
                .and_then(|s| s.strip_suffix(":streamGenerateContent"))
        })
        .ok_or_else(|| {
            (StatusCode::BAD_REQUEST, format!("Invalid Google GenAI path: {path}"))
        })?;
    let request = crate::protocol::google_genai::to_intermediate(body, model)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let result = state.engine.execute(&request).await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    let content = result.synthesis.unwrap_or_default();
    let ir = crate::protocol::IntermediateResponse {
        content,
        finish_reason: "STOP".into(),
        usage: Default::default(),
        cost_usd: result.total_cost_usd,
        duration_ms: result.duration_ms,
        model: state.config.fusion.name.clone(),
    };

    Ok(Json(crate::protocol::google_genai::from_intermediate(ir)))
}

async fn handle_anthropic_messages(
    State(state): State<AppState>,
    Json(body): Json<crate::protocol::anthropic_messages::MessagesRequest>,
) -> Result<Json<crate::protocol::anthropic_messages::MessagesResponse>, (StatusCode, String)> {
    tracing::debug!("Anthropic messages requested for model: {}", body.model);
    let model_name = state.config.fusion.name.clone();
    let request = crate::protocol::anthropic_messages::to_intermediate(body)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let result = state.engine.execute(&request).await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })?;

    let content = result.synthesis.unwrap_or_default();
    let ir = crate::protocol::IntermediateResponse {
        content,
        finish_reason: "end_turn".into(),
        usage: Default::default(),
        cost_usd: result.total_cost_usd,
        duration_ms: result.duration_ms,
        model: model_name.clone(),
    };

    Ok(Json(crate::protocol::anthropic_messages::from_intermediate(ir, &model_name)))
}
