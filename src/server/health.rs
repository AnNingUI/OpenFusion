//! Health check and metrics endpoints.

use axum::Json;
use serde::Serialize;
use std::sync::atomic::Ordering;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub name: &'static str,
}

pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        name: "openfusion",
    })
}

#[derive(Serialize)]
pub struct MetricsResponse {
    pub uptime_secs: u64,
    pub request_count: u64,
    pub fusion_name: String,
    pub worker_count: usize,
    pub judge_model: String,
}

pub async fn metrics(
    axum::extract::State(state): axum::extract::State<crate::server::AppState>,
) -> Json<MetricsResponse> {
    let uptime = state.start_time.elapsed().as_secs();
    let count = state.request_count.load(Ordering::Relaxed);
    Json(MetricsResponse {
        uptime_secs: uptime,
        request_count: count,
        fusion_name: state.config.fusion.name.clone(),
        worker_count: state.engine.workers().len(),
        judge_model: state.engine.judge().model.clone(),
    })
}
