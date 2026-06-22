//! OpenAI Chat Completions ↔ Intermediate format.
//!
//! POST /v1/chat/completions
//! Request:  { model, messages[{role,content}], max_tokens?, temperature?, stop? }
//! Response: { choices[{message:{content}, finish_reason}], usage, model }

use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, IntermediateResponse, Message, Role, Usage, WorkerResult};
use serde::{Deserialize, Serialize};

// ── Inbound request ──

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stop: Vec<String>,
    /// Passthrough: API key override from request body
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

fn default_max_tokens() -> u32 { 2048 }

// ── Outbound response ──

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Conversion ──

pub fn to_intermediate(req: ChatCompletionRequest) -> Result<IntermediateRequest, OpenFusionError> {
    // Validate
    if req.messages.is_empty() {
        return Err(OpenFusionError::Protocol("messages must not be empty".into()));
    }
    if req.max_tokens == 0 {
        return Err(OpenFusionError::Protocol("max_tokens must be > 0".into()));
    }
    if let Some(temp) = req.temperature
        && (!(0.0..=2.0).contains(&temp)) {
            return Err(OpenFusionError::Protocol(format!(
                "temperature must be 0.0..=2.0, got {temp}"
            )));
        }

    let messages = req.messages.into_iter().map(|m| {
        let role = match m.role.as_str() {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            _ => Role::User,
        };
        let content = match m.content {
            ChatContent::Text(t) => t,
            ChatContent::Parts(parts) => {
                parts.into_iter()
                    .filter(|p| p.kind == "text")
                    .filter_map(|p| p.text)
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };
        Message { role, content }
    }).collect();

    Ok(IntermediateRequest {
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stop: req.stop,
        api_key: req.api_key,
    })
}

pub fn from_intermediate(ir: IntermediateResponse, model_name: &str) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("openfusion-{}", uuid::Uuid::new_v4()),
        object: "chat.completion",
        created: chrono::Utc::now().timestamp(),
        model: model_name.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content: ir.content,
            },
            finish_reason: ir.finish_reason,
        }],
        usage: ChatUsage {
            prompt_tokens: ir.usage.prompt_tokens,
            completion_tokens: ir.usage.completion_tokens,
            total_tokens: ir.usage.total_tokens,
        },
    }
}

/// Build a raw chat completions request body for a specific worker model.
pub fn build_worker_body(ir: &IntermediateRequest, model: &str) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = ir.messages.iter().map(|m| {
        serde_json::json!({
            "role": match m.role {
                Role::System => "system",
                Role::Assistant => "assistant",
                Role::User => "user",
            },
            "content": m.content,
        })
    }).collect();

    serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": ir.max_tokens,
        "temperature": ir.temperature,
        "stop": ir.stop,
    })
}

/// Parse raw chat completion response body into IntermediateResponse.
pub fn parse_worker_response(
    body: &serde_json::Value,
    model: &str,
    duration_ms: u64,
    cost_usd: Option<f64>,
) -> Result<WorkerResult, OpenFusionError> {
    let choice = body["choices"].as_array()
        .and_then(|c| c.first())
        .ok_or_else(|| OpenFusionError::Protocol("missing choices[0]".into()))?;

    let content = choice["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let finish_reason = choice["finish_reason"]
        .as_str()
        .unwrap_or("stop")
        .to_string();

    let usage = Usage {
        prompt_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: body["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
    };

    Ok(WorkerResult {
        model: model.to_string(),
        api: "openai-completions".into(),
        success: true,
        response: Some(IntermediateResponse {
            content,
            finish_reason,
            usage,
            cost_usd: cost_usd.unwrap_or(0.0),
            duration_ms,
            model: model.to_string(),
        }),
        error: None,
    })
}
