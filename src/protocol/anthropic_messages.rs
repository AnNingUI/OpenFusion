//! Anthropic Messages API ↔ Intermediate format.
//!
//! POST /v1/messages
//! Request:  { model, system?, messages[{role,content}], max_tokens, temperature?, stop_sequences? }
//! Response: { id, content[{type,text}], stop_reason, model, usage }

use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, IntermediateResponse, Message, Role, Usage, WorkerResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub system: Option<AnthropicSystem>,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Structured(Vec<AnthropicSystemBlock>),
}

#[derive(Debug, Deserialize)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

// ── Outbound ──

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<AnthropicOutBlock>,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicOutBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ── Conversion ──

pub fn to_intermediate(req: MessagesRequest) -> Result<IntermediateRequest, OpenFusionError> {
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

    let mut messages = Vec::new();

    // Convert system
    match req.system {
        Some(AnthropicSystem::Text(t)) => {
            messages.push(Message { role: Role::System, content: t });
        }
        Some(AnthropicSystem::Structured(blocks)) => {
            let sys_text = blocks.into_iter()
                .filter(|b| b.kind == "text")
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n");
            messages.push(Message { role: Role::System, content: sys_text });
        }
        None => {}
    }

    for m in req.messages {
        let role = match m.role.as_str() {
            "assistant" => Role::Assistant,
            _ => Role::User,
        };
        let content = match m.content {
            AnthropicContent::Text(t) => t,
            AnthropicContent::Blocks(blocks) => {
                blocks.into_iter()
                    .filter(|b| b.kind == "text")
                    .filter_map(|b| b.text)
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };
        messages.push(Message { role, content });
    }

    Ok(IntermediateRequest {
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stop: req.stop_sequences,
        api_key: req.api_key,
    })
}

pub fn from_intermediate(ir: IntermediateResponse, model_name: &str) -> MessagesResponse {
    MessagesResponse {
        id: format!("openfusion-{}", uuid::Uuid::new_v4()),
        kind: "message",
        role: "assistant",
        model: model_name.to_string(),
        content: vec![AnthropicOutBlock {
            kind: "text",
            text: ir.content,
        }],
        stop_reason: ir.finish_reason,
        usage: AnthropicUsage {
            input_tokens: ir.usage.prompt_tokens,
            output_tokens: ir.usage.completion_tokens,
        },
    }
}

pub fn build_worker_body(ir: &IntermediateRequest, model: &str) -> serde_json::Value {
    let mut system_text = String::new();
    let mut messages: Vec<serde_json::Value> = Vec::new();

    for m in &ir.messages {
        match m.role {
            Role::System => {
                if !system_text.is_empty() {
                    system_text.push('\n');
                }
                system_text.push_str(&m.content);
            }
            role => {
                messages.push(serde_json::json!({
                    "role": match role {
                        Role::Assistant => "assistant",
                        _ => "user",
                    },
                    "content": m.content,
                }));
            }
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": ir.max_tokens,
        "temperature": ir.temperature,
        "stop_sequences": ir.stop,
    });

    if !system_text.is_empty() {
        body["system"] = serde_json::json!(system_text);
    }

    body
}

pub fn parse_worker_response(
    body: &serde_json::Value,
    model: &str,
    duration_ms: u64,
    cost_usd: Option<f64>,
) -> Result<WorkerResult, OpenFusionError> {
    let content = body["content"].as_array()
        .map(|blocks| {
            blocks.iter()
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    let stop_reason = body["stop_reason"]
        .as_str()
        .unwrap_or("end_turn")
        .to_string();

    Ok(WorkerResult {
        model: model.to_string(),
        api: "anthropic-messages".into(),
        success: true,
        response: Some(IntermediateResponse {
            content,
            finish_reason: stop_reason,
            usage: Usage {
                prompt_tokens: body["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: body["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: 0, // Anthropic doesn't give total
            },
            cost_usd: cost_usd.unwrap_or(0.0),
            duration_ms,
            model: model.to_string(),
        }),
        error: None,
    })
}
