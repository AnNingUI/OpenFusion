//! OpenAI Responses API ↔ Intermediate format.
//!
//! POST /v1/responses
//! Request:  { model, input: String|[{role,content}], instructions?, temperature?, ... }
//! Response: { output[{content[{text}]}], usage, model }

use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, IntermediateResponse, Message, Role, Usage, WorkerResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    pub instructions: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Messages(Vec<ResponsesMessage>),
}

#[derive(Debug, Deserialize)]
pub struct ResponsesMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ResponsesOut {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub model: String,
    pub output: Vec<ResponsesOutputItem>,
    pub usage: ResponsesUsage,
}

#[derive(Debug, Serialize)]
pub struct ResponsesOutputItem {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub content: Vec<ResponsesContent>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesContent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

fn default_max_tokens() -> u32 { 2048 }

// ── Conversion ──

pub fn to_intermediate(req: ResponsesRequest) -> Result<IntermediateRequest, OpenFusionError> {
    // Validate
    if req.max_output_tokens == 0 {
        return Err(OpenFusionError::Protocol("max_output_tokens must be > 0".into()));
    }
    if let Some(temp) = req.temperature
        && (!(0.0..=2.0).contains(&temp)) {
            return Err(OpenFusionError::Protocol(format!(
                "temperature must be 0.0..=2.0, got {temp}"
            )));
        }

    let mut messages = Vec::new();

    if let Some(instructions) = req.instructions {
        messages.push(Message { role: Role::System, content: instructions });
    }

    match req.input {
        ResponsesInput::Text(text) => {
            messages.push(Message { role: Role::User, content: text });
        }
        ResponsesInput::Messages(msgs) => {
            for m in msgs {
                let role = match m.role.as_str() {
                    "system" | "developer" => Role::System,
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                };
                messages.push(Message { role, content: m.content });
            }
        }
    }

    Ok(IntermediateRequest {
        messages,
        max_tokens: req.max_output_tokens,
        temperature: req.temperature,
        stop: vec![],
        api_key: req.api_key,
    })
}

pub fn from_intermediate(ir: IntermediateResponse, model_name: &str) -> ResponsesOut {
    ResponsesOut {
        id: format!("openfusion-{}", uuid::Uuid::new_v4()),
        object: "response",
        created_at: chrono::Utc::now().timestamp(),
        model: model_name.to_string(),
        output: vec![ResponsesOutputItem {
            kind: "message",
            content: vec![ResponsesContent {
                kind: "output_text",
                text: ir.content,
            }],
        }],
        usage: ResponsesUsage {
            input_tokens: ir.usage.prompt_tokens,
            output_tokens: ir.usage.completion_tokens,
            total_tokens: ir.usage.total_tokens,
        },
    }
}

pub fn build_worker_body(ir: &IntermediateRequest, model: &str) -> serde_json::Value {
    let input_messages: Vec<serde_json::Value> = ir.messages.iter().map(|m| {
        serde_json::json!({
            "role": match m.role {
                Role::System => "developer",
                Role::Assistant => "assistant",
                Role::User => "user",
            },
            "content": m.content,
        })
    }).collect();

    serde_json::json!({
        "model": model,
        "input": input_messages,
        "max_output_tokens": ir.max_tokens,
        "temperature": ir.temperature,
    })
}

pub fn parse_worker_response(
    body: &serde_json::Value,
    model: &str,
    duration_ms: u64,
    cost_usd: Option<f64>,
) -> Result<WorkerResult, OpenFusionError> {
    let output_items = body["output"].as_array()
        .ok_or_else(|| OpenFusionError::Protocol("missing output array".into()))?;

    let content = output_items.iter()
        .filter_map(|item| {
            item["content"].as_array().map(|contents| {
                contents.iter()
                    .filter_map(|c| c["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
        })
        .collect::<Vec<_>>()
        .join("\n");

    Ok(WorkerResult {
        model: model.to_string(),
        api: "openai-responses".into(),
        success: true,
        response: Some(IntermediateResponse {
            content,
            finish_reason: "stop".into(),
            usage: Usage {
                prompt_tokens: body["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: body["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: body["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
            },
            cost_usd: cost_usd.unwrap_or(0.0),
            duration_ms,
            model: model.to_string(),
        }),
        error: None,
    })
}
