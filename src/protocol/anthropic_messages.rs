//! Anthropic Messages API ↔ Intermediate format.
//!
//! POST /v1/messages
//! Request:  { model, system?, messages[{role,content}], max_tokens, temperature?, stop_sequences? }
//! Response: { id, content[{type,text}], stop_reason, model, usage }

use crate::error::OpenFusionError;
use crate::protocol::{
    IntermediateRequest, IntermediateResponse, Message, NATIVE_ANTHROPIC, Role, ThinkingConfig,
    ThinkingEffort, ToolCall, ToolDef, Usage, WorkerResult, merge_native_object, native_from_extra,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub system: Option<AnthropicSystem>,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub tools: Vec<AnthropicTool>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Inbound thinking config (if present)
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    /// Request streaming SSE response
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
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
    /// For tool_result blocks: the tool_use_id this result responds to
    pub tool_use_id: Option<String>,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub is_error: Option<bool>,
    /// For tool_use blocks: the tool name
    pub name: Option<String>,
    /// For tool_use blocks: the tool_use id
    pub id: Option<String>,
    /// For tool_use blocks: the tool input arguments
    #[serde(default)]
    pub input: Option<serde_json::Value>,
}

fn flatten_tool_result_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(serde_json::Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| match block {
                serde_json::Value::String(text) => Some(text.clone()),
                serde_json::Value::Object(object) => object
                    .get("text")
                    .and_then(|value| value.as_str())
                    .map(String::from)
                    .or_else(|| object.get("content").map(serde_json::Value::to_string))
                    .or_else(|| Some(serde_json::Value::Object(object.clone()).to_string())),
                value => Some(value.to_string()),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(value) => value.to_string(),
        None => String::new(),
    }
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
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
}

impl AnthropicOutBlock {
    fn text(text: String) -> Self {
        Self {
            kind: "text".into(),
            text: Some(text),
            thinking: None,
            id: None,
            name: None,
            input: None,
        }
    }

    fn thinking(thinking: String) -> Self {
        Self {
            kind: "thinking".into(),
            text: None,
            thinking: Some(thinking),
            id: None,
            name: None,
            input: None,
        }
    }

    fn tool_use(id: String, name: String, input: serde_json::Value) -> Self {
        Self {
            kind: "tool_use".into(),
            text: None,
            thinking: None,
            id: Some(id),
            name: Some(name),
            input: Some(input),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ── Conversion ──

fn effort_to_anthropic_str(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Disabled => "disabled",
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::Max => "max",
        ThinkingEffort::Adaptive => "adaptive",
    }
}

fn map_tool_choice_to_anthropic(tool_choice: &serde_json::Value) -> serde_json::Value {
    match tool_choice {
        serde_json::Value::String(choice) => match choice.as_str() {
            "none" => serde_json::json!({"type": "none"}),
            "required" | "any" => serde_json::json!({"type": "any"}),
            _ => serde_json::json!({"type": "auto"}),
        },
        serde_json::Value::Object(object) => {
            if object.contains_key("type")
                && object
                    .get("type")
                    .and_then(|value| value.as_str())
                    .is_some_and(|kind| matches!(kind, "auto" | "any" | "none" | "tool"))
            {
                return tool_choice.clone();
            }
            let name = object
                .get("name")
                .or_else(|| {
                    object
                        .get("function")
                        .and_then(|function| function.get("name"))
                })
                .and_then(|value| value.as_str())
                .unwrap_or("");
            if name.is_empty() {
                serde_json::json!({"type": "auto"})
            } else {
                serde_json::json!({"type": "tool", "name": name})
            }
        }
        _ => serde_json::json!({"type": "auto"}),
    }
}

pub fn to_intermediate(req: MessagesRequest) -> Result<IntermediateRequest, OpenFusionError> {
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

    let mut messages = Vec::new();

    // Convert system
    let mut system_text_for_ir: Option<String> = None;
    match req.system {
        Some(AnthropicSystem::Text(t)) => {
            system_text_for_ir = Some(t.clone());
            messages.push(Message {
                role: Role::System,
                content: t,
                tool_call_id: None,
                tool_is_error: None,
                tool_calls: vec![],
            });
        }
        Some(AnthropicSystem::Structured(blocks)) => {
            let sys_text = blocks
                .into_iter()
                .filter(|b| b.kind == "text")
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n");
            if !sys_text.is_empty() {
                system_text_for_ir = Some(sys_text.clone());
            }
            messages.push(Message {
                role: Role::System,
                content: sys_text,
                tool_call_id: None,
                tool_is_error: None,
                tool_calls: vec![],
            });
        }
        None => {}
    }

    for m in req.messages {
        match m.content {
            AnthropicContent::Text(t) => {
                let role = match m.role.as_str() {
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                };
                messages.push(Message {
                    role,
                    content: t,
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: vec![],
                });
            }
            AnthropicContent::Blocks(blocks) => {
                // Check if this is a tool_result message (upstream Claude Code executing a tool)
                let tool_results: Vec<_> =
                    blocks.iter().filter(|b| b.kind == "tool_result").collect();
                if !tool_results.is_empty() {
                    for block in tool_results {
                        let tid = block.tool_use_id.clone().unwrap_or_default();
                        let content_str = flatten_tool_result_content(block.content.as_ref());
                        messages.push(Message {
                            role: Role::Tool,
                            content: content_str,
                            tool_call_id: Some(tid),
                            tool_is_error: block.is_error,
                            tool_calls: vec![],
                        });
                    }
                    continue;
                }

                // Regular content blocks (text + tool_use from assistant)
                let text_parts: Vec<String> = blocks
                    .iter()
                    .filter(|b| b.kind == "text")
                    .filter_map(|b| b.text.clone())
                    .collect();
                let has_tool_use = blocks.iter().any(|b| b.kind == "tool_use");

                let role = if has_tool_use {
                    Role::Assistant
                } else {
                    match m.role.as_str() {
                        "assistant" => Role::Assistant,
                        _ => Role::User,
                    }
                };

                let content = if !text_parts.is_empty() {
                    text_parts.join("\n")
                } else {
                    String::new()
                };

                // Extract tool_use blocks for round-trip fidelity
                let tool_calls: Vec<crate::protocol::ToolCall> = blocks
                    .iter()
                    .filter(|b| b.kind == "tool_use")
                    .map(|b| crate::protocol::ToolCall {
                        id: b.id.clone().unwrap_or_default(),
                        name: b.name.clone().unwrap_or_default(),
                        arguments: b
                            .input
                            .clone()
                            .unwrap_or(serde_json::Value::Object(Default::default())),
                    })
                    .collect();

                messages.push(Message {
                    role,
                    content,
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls,
                });
            }
        }
    }

    // Convert Anthropic tools → ToolDef
    let tools: Vec<ToolDef> = req
        .tools
        .into_iter()
        .map(|t| ToolDef {
            name: t.name,
            description: t.description.unwrap_or_default(),
            parameters: t.input_schema,
        })
        .collect();

    // Preserve thinking config from inbound request if present
    let thinking = req.thinking.map(|v| {
        // Try to extract effort from the thinking block
        // Anthropic inbound: { "type": "enabled"/"disabled"/"auto", "budget_tokens": ... }
        let effort = match v.get("type").and_then(|t| t.as_str()) {
            Some("disabled") => ThinkingEffort::Disabled,
            Some("auto") | Some("adaptive") => ThinkingEffort::Adaptive,
            _ => {
                // Try budget_tokens to infer effort
                match v.get("budget_tokens").and_then(|b| b.as_i64()) {
                    Some(0) => ThinkingEffort::Disabled,
                    _ => ThinkingEffort::High,
                }
            }
        };
        ThinkingConfig {
            effort,
            budget_tokens: v
                .get("budget_tokens")
                .and_then(|b| b.as_i64())
                .map(|n| n as i32),
            summary: false,
        }
    });

    Ok(IntermediateRequest {
        messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences,
        api_key: req.api_key,
        tools,
        tool_choice: req.tool_choice,
        parallel_tool_calls: None,
        thinking,
        session_id: None,
        system: system_text_for_ir,
        stream: req.stream,
        metadata: req.metadata,
        native: native_from_extra(NATIVE_ANTHROPIC, req.extra),
    })
}

pub fn from_intermediate(ir: IntermediateResponse, model_name: &str) -> MessagesResponse {
    let mut content: Vec<AnthropicOutBlock> = Vec::new();

    // Add thinking (must come before text)
    if let Some(ref thinking) = ir.thinking
        && !thinking.is_empty()
    {
        content.push(AnthropicOutBlock::thinking(thinking.clone()));
    }

    // Add text
    if !ir.content.is_empty() {
        content.push(AnthropicOutBlock::text(ir.content));
    }

    // Add tool_use blocks
    for tc in &ir.tool_calls {
        let input = match &tc.arguments {
            serde_json::Value::Object(map) => serde_json::Value::Object(map.clone()),
            serde_json::Value::String(s) => {
                serde_json::from_str(s).unwrap_or(serde_json::Value::Object(Default::default()))
            }
            _ => serde_json::Value::Object(Default::default()),
        };
        content.push(AnthropicOutBlock::tool_use(
            tc.id.clone(),
            tc.name.clone(),
            input,
        ));
    }

    MessagesResponse {
        id: format!("openfusion-{}", uuid::Uuid::new_v4()),
        kind: "message",
        role: "assistant",
        model: model_name.to_string(),
        content,
        stop_reason: if ir.tool_calls.is_empty() {
            ir.finish_reason.clone()
        } else {
            "tool_use".into()
        },
        usage: AnthropicUsage {
            input_tokens: ir.usage.prompt_tokens,
            output_tokens: ir.usage.completion_tokens,
        },
    }
}

pub fn build_worker_body(ir: &IntermediateRequest, model: &str) -> serde_json::Value {
    let mut system_text = String::new();

    // Start with native system field (highest priority)
    if let Some(ref sys) = ir.system {
        system_text.push_str(sys);
    }

    let mut messages: Vec<serde_json::Value> = Vec::new();

    for m in &ir.messages {
        match m.role {
            Role::System => {
                if !system_text.is_empty() {
                    system_text.push('\n');
                }
                system_text.push_str(&m.content);
            }
            Role::Tool => {
                // Tool results as user messages with tool_result content blocks
                let mut result = serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.as_deref().unwrap_or(""),
                        "content": m.content,
                    }],
                });
                if let Some(is_error) = m.tool_is_error {
                    result["content"][0]["is_error"] = serde_json::json!(is_error);
                }
                messages.push(result);
            }
            Role::Assistant => {
                // Build content blocks: text + tool_use (if any)
                let mut blocks: Vec<serde_json::Value> = Vec::new();
                if !m.content.is_empty() {
                    blocks.push(serde_json::json!({"type": "text", "text": m.content}));
                }
                for tc in &m.tool_calls {
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                let content = if blocks.len() == 1 && blocks[0]["type"] == "text" {
                    // Single text block → use plain string for compatibility
                    serde_json::Value::String(m.content.clone())
                } else if blocks.is_empty() {
                    serde_json::Value::String(String::new())
                } else {
                    serde_json::Value::Array(blocks)
                };
                messages.push(serde_json::json!({"role": "assistant", "content": content}));
            }
            _ => {
                messages.push(serde_json::json!({
                    "role": "user",
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

    if !system_text.is_empty() {
        body["system"] = serde_json::json!(system_text);
    }
    if let Some(temp) = ir.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if !ir.stop.is_empty() {
        body["stop_sequences"] = serde_json::json!(ir.stop);
    }

    // Inject tool definitions (Anthropic format: input_schema instead of parameters)
    if !ir.tools.is_empty() {
        let tools: Vec<serde_json::Value> = ir
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            })
            .collect();
        body["tools"] = serde_json::json!(tools);
    }
    if let Some(ref tool_choice) = ir.tool_choice {
        body["tool_choice"] = map_tool_choice_to_anthropic(tool_choice);
    }
    if let Some(top_p) = ir.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }

    // Inject thinking configuration
    if let Some(ref thinking) = ir.thinking {
        match thinking.effort {
            ThinkingEffort::Disabled => {
                body["thinking"] = serde_json::json!({"type": "disabled"});
            }
            ThinkingEffort::Adaptive => {
                body["thinking"] = serde_json::json!({"type": "adaptive"});
                let effort_str = effort_to_anthropic_str(thinking.effort);
                body["output_config"] = serde_json::json!({"effort": effort_str});
            }
            _ => {
                // Low / Medium / High / Max → enabled with budget
                let budget = thinking.budget_tokens.unwrap_or(-1);
                body["thinking"] = serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget,
                });
            }
        }
    }

    // Enable SSE streaming if requested
    if ir.stream {
        body["stream"] = serde_json::json!(true);
    }

    if !ir.metadata.is_empty() {
        body["metadata"] = serde_json::Value::Object(ir.metadata.clone());
    }
    merge_native_object(&mut body, &ir.native, NATIVE_ANTHROPIC);
    body
}

pub fn parse_worker_response(
    body: &serde_json::Value,
    model: &str,
    duration_ms: u64,
    cost_usd: Option<f64>,
) -> Result<WorkerResult, OpenFusionError> {
    let content = body["content"]
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    // Extract thinking text from content blocks where type == "thinking"
    let thinking = body["content"]
        .as_array()
        .map(|blocks| {
            let parts: Vec<&str> = blocks
                .iter()
                .filter(|b| b["type"].as_str() == Some("thinking"))
                .filter_map(|b| b["thinking"].as_str())
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(""))
            }
        })
        .unwrap_or(None);

    let stop_reason = body["stop_reason"]
        .as_str()
        .unwrap_or("end_turn")
        .to_string();

    // Extract tool_use blocks from content
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    if let Some(blocks) = body["content"].as_array() {
        for block in blocks {
            if block["type"].as_str() == Some("tool_use") {
                tool_calls.push(ToolCall {
                    id: block["id"].as_str().unwrap_or("").to_string(),
                    name: block["name"].as_str().unwrap_or("").to_string(),
                    arguments: block["input"].clone(),
                });
            }
        }
    }

    let native = crate::protocol::native_from_body_excluding(
        NATIVE_ANTHROPIC,
        body,
        &[
            "id",
            "type",
            "role",
            "model",
            "content",
            "stop_reason",
            "stop_sequence",
            "usage",
        ],
    );

    Ok(WorkerResult {
        model: model.to_string(),
        name: String::new(),
        api: "anthropic-messages".into(),
        success: true,
        response: Some(IntermediateResponse {
            id: body["id"].as_str().map(String::from),
            content,
            status: None,
            finish_reason: stop_reason,
            usage: Usage {
                prompt_tokens: body["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: body["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: 0, // Anthropic doesn't give total
                reasoning_tokens: 0,
            },
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
    fn anthropic_tool_result_preserves_is_error_and_flattens_content_blocks() {
        let req: MessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude",
            "max_tokens": 128,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "is_error": true,
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                        {"json": true}
                    ]
                }]
            }]
        }))
        .unwrap();

        let ir = to_intermediate(req).unwrap();
        let tool = ir.messages.iter().find(|m| m.role == Role::Tool).unwrap();
        assert_eq!(tool.tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(tool.tool_is_error, Some(true));
        assert_eq!(tool.content, "first\nsecond\n{\"json\":true}");

        let body = build_worker_body(&ir, "claude");
        assert_eq!(body["messages"][0]["content"][0]["is_error"], true);
        assert_eq!(body["messages"][0]["content"][0]["content"], tool.content);
    }

    #[test]
    fn anthropic_worker_body_omits_null_temperature_and_empty_stop_sequences() {
        let ir = IntermediateRequest {
            messages: vec![Message {
                role: Role::User,
                content: "hello".into(),
                tool_call_id: None,
                tool_is_error: None,
                tool_calls: vec![],
            }],
            max_tokens: 128,
            temperature: None,
            top_p: None,
            stop: vec![],
            api_key: None,
            tools: vec![],
            tool_choice: None,
            parallel_tool_calls: None,
            thinking: None,
            session_id: None,
            system: None,
            stream: false,
            metadata: Default::default(),
            native: Default::default(),
        };

        let body = build_worker_body(&ir, "claude");
        assert!(body.get("temperature").is_none());
        assert!(body.get("stop_sequences").is_none());
    }
}
