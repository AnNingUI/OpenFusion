//! fusion tool: advisory worker panel with optional judge synthesis.

use crate::engine::{FusionEngine, FusionResult};
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, Message, Role};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
pub struct FusionParams {
    pub prompt: String,
    pub system: Option<String>,
    pub judge_mode: Option<bool>,
    pub panel_override: Option<Vec<String>>,
    pub save_session: Option<bool>,
}

pub async fn run(
    engine: Arc<FusionEngine>,
    params: FusionParams,
) -> Result<FusionResult, OpenFusionError> {
    let prompt = match params
        .system
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(system) => format!("# Domain Framing\n{system}\n\n# Task\n{}", params.prompt),
        None => params.prompt.clone(),
    };
    let messages = vec![Message {
        role: Role::User,
        content: prompt,
        tool_call_id: None,
        tool_is_error: None,
        tool_calls: vec![],
    }];

    let request = IntermediateRequest {
        messages,
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

    let model_filter = params.panel_override.unwrap_or_default();

    engine
        .execute_advisory(
            &request,
            &model_filter,
            params.judge_mode.unwrap_or(false),
            params.save_session.unwrap_or(false),
        )
        .await
}
