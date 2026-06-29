//! fusion_diff tool: raw worker comparison without semantic synthesis.

use crate::engine::{FusionEngine, FusionResult};
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, Message, Role};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
pub struct DiffParams {
    pub prompt: String,
    pub system: Option<String>,
    pub panel_override: Option<Vec<String>>,
    pub save_session: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct DiffResult {
    pub session_id: String,
    pub responses: Vec<WorkerResponseView>,
    pub summary: DiffSummary,
    pub cost_usd: f64,
    pub duration_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct WorkerResponseView {
    pub name: String,
    pub model: String,
    pub api: String,
    pub content: String,
    pub token_count: u32,
    pub duration_ms: u64,
    pub cost_usd: f64,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DiffSummary {
    pub mode: String,
    pub note: String,
    pub models_succeeded: usize,
    pub models_failed: usize,
}

pub async fn run(
    engine: Arc<FusionEngine>,
    params: DiffParams,
) -> Result<DiffResult, OpenFusionError> {
    let prompt = match params
        .system
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(system) => format!("# Domain Framing\n{system}\n\n# Task\n{}", params.prompt),
        None => params.prompt.clone(),
    };
    let request = IntermediateRequest {
        messages: vec![Message {
            role: Role::User,
            content: prompt,
            tool_call_id: None,
            tool_is_error: None,
            tool_calls: vec![],
        }],
        max_tokens: 2048,
        temperature: Some(0.7),
        top_p: None,
        stop: vec![],
        api_key: None,
        tools: Vec::new(),
        tool_choice: None,
        parallel_tool_calls: None,
        thinking: None,
        session_id: None,
        stream: false,
        system: None,
        metadata: Default::default(),
        native: Default::default(),
    };

    let start = std::time::Instant::now();
    let model_filter = params.panel_override.unwrap_or_default();
    let worker_results = engine
        .execute_workers_filtered(&request, &model_filter)
        .await;
    if worker_results.is_empty() {
        return Err(OpenFusionError::Config(
            "No workers matched panel_override".into(),
        ));
    }
    let duration_ms = start.elapsed().as_millis() as u64;
    let session_id = uuid::Uuid::new_v4().to_string();

    let responses = worker_results
        .iter()
        .map(|wr| {
            let response = wr.response.as_ref();
            WorkerResponseView {
                name: wr.name.clone(),
                model: wr.model.clone(),
                api: wr.api.clone(),
                content: response.map(|r| r.content.clone()).unwrap_or_default(),
                token_count: response.map(|r| r.usage.total_tokens).unwrap_or(0),
                duration_ms: response.map(|r| r.duration_ms).unwrap_or(0),
                cost_usd: response.map(|r| r.cost_usd).unwrap_or(0.0),
                error: wr.error.clone(),
            }
        })
        .collect::<Vec<_>>();
    let total_cost = responses.iter().map(|r| r.cost_usd).sum::<f64>();
    let models_succeeded = responses.iter().filter(|r| r.error.is_none()).count();
    let models_failed = responses.len().saturating_sub(models_succeeded);

    if params.save_session.unwrap_or(false) {
        let original_prompt = request
            .messages
            .iter()
            .filter(|message| matches!(message.role, Role::User))
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let result = FusionResult {
            session_id: session_id.clone(),
            judge_response: None,
            worker_results,
            enhanced_prompt_used: false,
            synthesis: None,
            consensus: vec![],
            contradictions: vec![],
            blind_spots: vec![],
            judge_output_raw: None,
            passthrough_tool_calls: vec![],
            total_cost_usd: total_cost,
            models_succeeded,
            models_failed,
            duration_ms,
            timestamp: chrono::Utc::now().to_rfc3339(),
            original_prompt,
        };
        let _ = engine.sessions().save(&result).await;
    }

    Ok(DiffResult {
        session_id,
        responses,
        summary: DiffSummary {
            mode: "raw-worker-comparison".into(),
            note: "No semantic consensus or disagreement inference is performed. The calling AI client should compare these worker outputs or call fusion with judge_mode=true."
                .into(),
            models_succeeded,
            models_failed,
        },
        cost_usd: total_cost,
        duration_ms,
    })
}
