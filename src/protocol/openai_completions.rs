//! OpenAI Chat Completions 鈫?Intermediate format.
//!
//! POST /v1/chat/completions
//! Request:  { model, messages[{role,content}], max_tokens?, temperature?, stop? }
//! Response: { choices[{message:{content}, finish_reason}], usage, model }

use crate::error::OpenFusionError;
use crate::protocol::{
    IntermediateRequest, IntermediateResponse, Message, NATIVE_OPENAI_CHAT, Role, ThinkingConfig,
    ThinkingEffort, ToolCall, Usage, WorkerResult, merge_native_object, native_from_extra,
};
use serde::{Deserialize, Serialize};

// 鈹€鈹€ Inbound request 鈹€鈹€

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Vec<String>,
    /// Passthrough: API key override from request body
    #[serde(default)]
    pub api_key: Option<String>,
    /// Request streaming SSE response
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<ChatTool>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<ChatReasoning>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ChatReasoning {
    #[serde(default)]
    pub effort: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<ChatContent>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub is_error: Option<bool>,
    #[serde(default)]
    pub tool_calls: Vec<ChatToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatToolFunction,
}

#[derive(Debug, Deserialize)]
pub struct ChatToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(default)]
    pub function: ChatToolCallFunction,
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatToolCallFunction {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for ChatContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

fn default_max_tokens() -> u32 {
    2048
}

fn effort_to_openai_str(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Disabled => "none",
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::Max => "xhigh",
        ThinkingEffort::Adaptive => "medium",
    }
}

fn parse_openai_chat_reasoning(reasoning: Option<ChatReasoning>) -> Option<ThinkingConfig> {
    let reasoning = reasoning?;
    let effort = match reasoning.effort.as_deref() {
        Some("none" | "disabled") => ThinkingEffort::Disabled,
        Some("minimal" | "low") => ThinkingEffort::Low,
        Some("medium") => ThinkingEffort::Medium,
        Some("high") => ThinkingEffort::High,
        Some("xhigh" | "max" | "ultra") => ThinkingEffort::Max,
        Some(_) | None => ThinkingEffort::Adaptive,
    };
    Some(ThinkingConfig {
        effort,
        budget_tokens: None,
        summary: false,
    })
}

// 鈹€鈹€ Outbound response 鈹€鈹€

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
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ChatResponseToolCall>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ChatResponseToolCallFunction,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// 鈹€鈹€ Conversion 鈹€鈹€

pub fn to_intermediate(req: ChatCompletionRequest) -> Result<IntermediateRequest, OpenFusionError> {
    // Validate
    if req.messages.is_empty() {
        return Err(OpenFusionError::Protocol(
            "messages must not be empty".into(),
        ));
    }
    if req.max_tokens == 0 {
        return Err(OpenFusionError::Protocol("max_tokens must be > 0".into()));
    }
    if let Some(temp) = req.temperature
        && (!(0.0..=2.0).contains(&temp))
    {
        return Err(OpenFusionError::Protocol(format!(
            "temperature must be 0.0..=2.0, got {temp}"
        )));
    }

    let messages = req
        .messages
        .into_iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => Role::User,
            };
            let content = match m.content.unwrap_or_default() {
                ChatContent::Text(t) => t,
                ChatContent::Parts(parts) => parts
                    .into_iter()
                    .filter(|p| p.kind == "text")
                    .filter_map(|p| p.text)
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            let tool_calls = m
                .tool_calls
                .into_iter()
                .map(|tc| {
                    let arguments =
                        serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                            .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                    ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments,
                    }
                })
                .collect();
            Message {
                role,
                content,
                tool_call_id: m.tool_call_id,
                tool_is_error: m.is_error,
                tool_calls,
            }
        })
        .collect();

    let tools = req
        .tools
        .into_iter()
        .filter(|t| t.kind == "function")
        .map(|t| crate::protocol::ToolDef {
            name: t.function.name,
            description: t.function.description,
            parameters: t.function.parameters,
        })
        .collect();

    let thinking = parse_openai_chat_reasoning(req.reasoning);

    Ok(IntermediateRequest {
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop,
        api_key: req.api_key,
        tools,
        tool_choice: req.tool_choice,
        parallel_tool_calls: req.parallel_tool_calls,
        thinking,
        session_id: None,
        system: None,
        stream: req.stream,
        metadata: req.metadata,
        native: native_from_extra(NATIVE_OPENAI_CHAT, req.extra),
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
                content: if ir.content.is_empty() {
                    None
                } else {
                    Some(ir.content)
                },
                reasoning: ir.thinking,
                tool_calls: ir
                    .tool_calls
                    .into_iter()
                    .map(|tc| ChatResponseToolCall {
                        id: tc.id,
                        kind: "function",
                        function: ChatResponseToolCallFunction {
                            name: tc.name,
                            arguments: tc.arguments.to_string(),
                        },
                    })
                    .collect(),
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
    let mut messages: Vec<serde_json::Value> = Vec::new();

    for m in &ir.messages {
        match m.role {
            Role::Tool => {
                // Tool results must follow an assistant message with tool_calls
                let mut msg = serde_json::json!({
                    "role": "tool",
                    "content": m.content,
                });
                if let Some(ref call_id) = m.tool_call_id {
                    msg["tool_call_id"] = serde_json::json!(call_id);
                }
                if let Some(is_error) = m.tool_is_error {
                    msg["is_error"] = serde_json::json!(is_error);
                }
                messages.push(msg);
            }
            Role::Assistant if !m.tool_calls.is_empty() => {
                // Assistant with tool_calls 鈥?must include tool_calls field
                let tcs: Vec<serde_json::Value> = m
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments.to_string(),
                            }
                        })
                    })
                    .collect();
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": if m.content.is_empty() { serde_json::Value::Null } else { serde_json::json!(m.content) },
                    "tool_calls": tcs,
                }));
            }
            _ => {
                // Regular system/user/assistant message
                let role = match m.role {
                    Role::System => "system",
                    Role::Assistant => "assistant",
                    Role::User => "user",
                    Role::Tool => "tool", // unreachable but fallback
                };
                messages.push(serde_json::json!({
                    "role": role,
                    "content": m.content,
                }));
            }
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": ir.max_tokens,
    });

    if let Some(temp) = ir.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(top_p) = ir.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if !ir.stop.is_empty() {
        body["stop"] = serde_json::json!(ir.stop);
    }

    // Inject tool definitions
    if !ir.tools.is_empty() {
        let tools: Vec<serde_json::Value> = ir
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body["tools"] = serde_json::json!(tools);
    }
    if let Some(ref tool_choice) = ir.tool_choice {
        body["tool_choice"] = tool_choice.clone();
    }
    if let Some(parallel_tool_calls) = ir.parallel_tool_calls {
        body["parallel_tool_calls"] = serde_json::json!(parallel_tool_calls);
    }
    if !ir.metadata.is_empty() {
        body["metadata"] = serde_json::Value::Object(ir.metadata.clone());
    }

    // Inject reasoning/thinking configuration
    if let Some(ref thinking) = ir.thinking {
        let effort_str = effort_to_openai_str(thinking.effort);
        body["reasoning"] = serde_json::json!({"effort": effort_str});
    }

    // Enable SSE streaming if requested
    if ir.stream {
        body["stream"] = serde_json::json!(true);
    }

    merge_native_object(&mut body, &ir.native, NATIVE_OPENAI_CHAT);
    body
}

/// Parse raw chat completion response body into IntermediateResponse.
pub fn parse_worker_response(
    body: &serde_json::Value,
    model: &str,
    duration_ms: u64,
    cost_usd: Option<f64>,
) -> Result<WorkerResult, OpenFusionError> {
    let choice = body["choices"]
        .as_array()
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

    let mut usage = Usage {
        prompt_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: body["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
        reasoning_tokens: 0,
    };

    // Extract reasoning_tokens from output_tokens_details if present
    if let Some(reasoning_tokens) =
        body["usage"]["output_tokens_details"]["reasoning_tokens"].as_u64()
    {
        usage.reasoning_tokens = reasoning_tokens as u32;
    }

    // Extract tool calls if present
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    if let Some(tcs) = choice["message"]["tool_calls"].as_array() {
        for tc in tcs {
            tool_calls.push(ToolCall {
                id: tc["id"].as_str().unwrap_or("").to_string(),
                name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                arguments: match serde_json::from_str::<serde_json::Value>(
                    tc["function"]["arguments"].as_str().unwrap_or("{}"),
                ) {
                    Ok(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
                    Ok(serde_json::Value::String(s)) => serde_json::from_str(&s)
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                    _ => serde_json::Value::Object(Default::default()),
                },
            });
        }
    }

    // Extract reasoning/thinking if present (reasoning models like Sensenova)
    let thinking = choice["message"]["reasoning"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(String::from);

    let native = crate::protocol::native_from_body_excluding(
        NATIVE_OPENAI_CHAT,
        body,
        &["id", "object", "created", "model", "choices", "usage"],
    );

    Ok(WorkerResult {
        model: model.to_string(),
        name: String::new(),
        api: "openai-completions".into(),
        success: true,
        response: Some(IntermediateResponse {
            id: body["id"].as_str().map(String::from),
            content,
            status: None,
            finish_reason,
            usage,
            cost_usd: cost_usd.unwrap_or(0.0),
            duration_ms,
            model: model.to_string(),
            tool_calls,
            thinking,
            metadata: Default::default(),
            native,
        }),
        error: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_request_preserves_tools_and_tool_history() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "demo",
            "messages": [
                {"role": "user", "content": "read it"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "inspect_input", "arguments": "{\"path\":\"sample-target\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "inspect_input", "description": "read", "parameters": {"type": "object"}}
            }]
        })).unwrap();

        let ir = to_intermediate(req).unwrap();
        assert_eq!(ir.tools.len(), 1);
        assert_eq!(ir.messages[1].role, Role::Assistant);
        assert_eq!(ir.messages[1].tool_calls[0].id, "call_1");
        assert_eq!(ir.messages[2].role, Role::Tool);
        assert_eq!(ir.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn chat_request_preserves_reasoning_config() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "demo",
            "messages": [{"role": "user", "content": "think"}],
            "reasoning": {"effort": "max"}
        }))
        .unwrap();

        let ir = to_intermediate(req).unwrap();
        let thinking = ir.thinking.expect("reasoning should map to thinking");
        assert_eq!(thinking.effort, ThinkingEffort::Max);
        assert!(!thinking.summary);
    }
}
