//! OpenAI Responses API ↔ Intermediate format.
//!
//! POST /v1/responses
//! Request:  { model, input: String|[{role,content}], instructions?, temperature?, ... }
//! Response: { output[{content[{text}]}], usage, model }

use crate::error::OpenFusionError;
use crate::protocol::{
    IntermediateRequest, IntermediateResponse, Message, NATIVE_OPENAI_RESPONSES, Role,
    ThinkingConfig, ThinkingEffort, ToolCall, Usage, WorkerResult, merge_native_object,
    native_from_extra,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    #[allow(dead_code)]
    pub model: String,
    pub input: ResponsesInput,
    pub instructions: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Request streaming SSE response
    #[serde(default)]
    pub stream: bool,
    /// Session continuity: resume from a previous response
    #[serde(default)]
    #[allow(dead_code)]
    pub previous_response_id: Option<String>,
    /// Tool definitions (Codex sends function tools)
    #[serde(default)]
    pub tools: Vec<ResponsesTool>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI Responses reasoning configuration.
    #[serde(default)]
    pub reasoning: Option<ResponsesReasoning>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesReasoning {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub summary: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub function: Option<ResponsesToolFunction>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesToolFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        #[serde(default)]
        content: ResponsesMessageContent,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        #[serde(default)]
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        #[serde(default)]
        output: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesMessageContent {
    Text(String),
    Blocks(Vec<ResponsesContentBlock>),
}

#[derive(Debug, Deserialize)]
pub struct ResponsesContentBlock {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub text: Option<String>,
}

impl Default for ResponsesMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize)]
pub struct ResponsesOut {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub model: String,
    pub status: String,
    pub output: Vec<ResponsesOutputItem>,
    pub usage: ResponsesUsage,
}

#[derive(Debug, Serialize)]
pub struct ResponsesOutputItem {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    pub content: Vec<ResponsesContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesContent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
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

// ── Conversion ──

fn parse_openai_reasoning(reasoning: Option<ResponsesReasoning>) -> Option<ThinkingConfig> {
    let reasoning = reasoning?;
    let effort = match reasoning.effort.as_deref() {
        Some("none" | "disabled") => ThinkingEffort::Disabled,
        Some("minimal" | "low") => ThinkingEffort::Low,
        Some("medium") => ThinkingEffort::Medium,
        Some("high") => ThinkingEffort::High,
        Some("xhigh" | "max" | "ultra") => ThinkingEffort::Max,
        Some(_) | None => ThinkingEffort::Adaptive,
    };
    let summary = match reasoning.summary {
        Some(serde_json::Value::Null) | None => false,
        Some(serde_json::Value::String(s)) => !matches!(s.as_str(), "none" | "off" | "disabled"),
        Some(serde_json::Value::Bool(enabled)) => enabled,
        Some(_) => true,
    };
    Some(ThinkingConfig {
        effort,
        budget_tokens: None,
        summary,
    })
}

fn dedupe_repeated_context_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut seen_system = std::collections::HashSet::new();
    let mut seen_environment_context = std::collections::HashSet::new();
    let mut deduped = Vec::with_capacity(messages.len());

    for message in messages {
        match message.role {
            Role::System if !seen_system.insert(message.content.clone()) => continue,
            Role::User
                if message
                    .content
                    .trim_start()
                    .starts_with("<environment_context>")
                    && !seen_environment_context.insert(message.content.clone()) =>
            {
                continue;
            }
            _ => {}
        }
        deduped.push(message);
    }

    deduped
}

pub fn to_intermediate(req: ResponsesRequest) -> Result<IntermediateRequest, OpenFusionError> {
    // Validate
    if req.max_output_tokens == 0 {
        return Err(OpenFusionError::Protocol(
            "max_output_tokens must be > 0".into(),
        ));
    }
    if let Some(temp) = req.temperature
        && (!(0.0..=2.0).contains(&temp))
    {
        return Err(OpenFusionError::Protocol(format!(
            "temperature must be 0.0..=2.0, got {temp}"
        )));
    }

    // Debug: log raw input structure before conversion
    tracing::info!(
        "[RESPONSES-TO-IR] input type: {}",
        match &req.input {
            ResponsesInput::Text(t) => format!("Text({} chars)", t.len()),
            ResponsesInput::Items(items) => format!("Items({} items)", items.len()),
        }
    );
    if let ResponsesInput::Items(items) = &req.input {
        for (i, item) in items.iter().enumerate() {
            match item {
                ResponsesInputItem::Message { role, content } => {
                    let content_debug = match content {
                        ResponsesMessageContent::Text(t) => format!("Text({} chars)", t.len()),
                        ResponsesMessageContent::Blocks(blocks) => format!(
                            "Blocks({} blocks, types={:?})",
                            blocks.len(),
                            blocks
                                .iter()
                                .filter_map(|b| b.kind.as_deref())
                                .collect::<Vec<_>>()
                        ),
                    };
                    tracing::info!(
                        "[RESPONSES-TO-IR] item[{}] type=message role={} content={}",
                        i,
                        role,
                        content_debug
                    );
                }
                ResponsesInputItem::FunctionCall { call_id, name, .. } => {
                    tracing::info!(
                        "[RESPONSES-TO-IR] item[{}] type=function_call call_id={} name={}",
                        i,
                        call_id,
                        name
                    );
                }
                ResponsesInputItem::FunctionCallOutput { call_id, .. } => {
                    tracing::info!(
                        "[RESPONSES-TO-IR] item[{}] type=function_call_output call_id={}",
                        i,
                        call_id
                    );
                }
            }
        }
    }

    let mut messages = Vec::new();
    let thinking = parse_openai_reasoning(req.reasoning);

    let system = req.instructions.clone();
    if let Some(instructions) = req.instructions {
        messages.push(Message {
            role: Role::System,
            content: instructions,
            tool_call_id: None,
            tool_is_error: None,
            tool_calls: vec![],
        });
    }

    match req.input {
        ResponsesInput::Text(text) => {
            messages.push(Message {
                role: Role::User,
                content: text,
                tool_call_id: None,
                tool_is_error: None,
                tool_calls: vec![],
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                match item {
                    ResponsesInputItem::Message { role, content } => {
                        let r = match role.as_str() {
                            "system" | "developer" => Role::System,
                            "assistant" => Role::Assistant,
                            _ => Role::User,
                        };
                        let c = match content {
                            ResponsesMessageContent::Text(t) => t,
                            ResponsesMessageContent::Blocks(blocks) => blocks
                                .into_iter()
                                .filter(|b| {
                                    matches!(
                                        b.kind.as_deref(),
                                        Some("text") | Some("input_text") | Some("output_text")
                                    )
                                })
                                .filter_map(|b| b.text)
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        messages.push(Message {
                            role: r,
                            content: c,
                            tool_call_id: None,
                            tool_is_error: None,
                            tool_calls: vec![],
                        });
                    }
                    ResponsesInputItem::FunctionCall {
                        call_id,
                        name,
                        arguments,
                    } => {
                        let args: serde_json::Value = serde_json::from_str(&arguments)
                            .unwrap_or(serde_json::Value::Object(Default::default()));
                        // Attach to previous assistant message or create one
                        if let Some(last) = messages.last_mut()
                            && last.role == Role::Assistant
                        {
                            last.tool_calls.push(ToolCall {
                                id: call_id,
                                name,
                                arguments: args,
                            });
                            continue;
                        }
                        messages.push(Message {
                            role: Role::Assistant,
                            content: String::new(),
                            tool_call_id: None,
                            tool_is_error: None,
                            tool_calls: vec![ToolCall {
                                id: call_id,
                                name,
                                arguments: args,
                            }],
                        });
                    }
                    ResponsesInputItem::FunctionCallOutput { call_id, output } => {
                        messages.push(Message {
                            role: Role::Tool,
                            content: output,
                            tool_call_id: Some(call_id),
                            tool_is_error: None,
                            tool_calls: vec![],
                        });
                    }
                }
            }
        }
    }

    // Convert tools from Responses format to ToolDef
    let tools: Vec<crate::protocol::ToolDef> = req
        .tools
        .into_iter()
        .filter(|t| t.kind == "function")
        .filter_map(|t| {
            let (name, description, parameters) = if let Some(function) = t.function {
                (function.name, function.description, function.parameters)
            } else {
                (
                    t.name,
                    t.description,
                    if t.parameters.is_null() {
                        t.input_schema
                    } else {
                        t.parameters
                    },
                )
            };
            if name.is_empty() {
                return None;
            }
            Some(crate::protocol::ToolDef {
                name,
                description,
                parameters,
            })
        })
        .collect();

    Ok(IntermediateRequest {
        messages: dedupe_repeated_context_messages(messages),
        max_tokens: req.max_output_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: vec![],
        api_key: req.api_key,
        tools,
        tool_choice: req.tool_choice,
        parallel_tool_calls: req.parallel_tool_calls,
        thinking,
        session_id: req.previous_response_id,
        system,
        stream: req.stream,
        metadata: req.metadata,
        native: native_from_extra(NATIVE_OPENAI_RESPONSES, req.extra),
    })
}

pub fn from_intermediate(ir: IntermediateResponse, model_name: &str) -> ResponsesOut {
    let mut output = Vec::new();
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4());

    // Add reasoning item if thinking is present
    if let Some(ref thinking) = ir.thinking
        && !thinking.is_empty()
    {
        output.push(ResponsesOutputItem {
            id: format!("rs_{}", uuid::Uuid::new_v4()),
            kind: "reasoning",
            status: "completed".into(),
            role: None,
            content: vec![ResponsesContent {
                kind: "reasoning_text",
                text: Some(thinking.clone()),
                summary: None,
            }],
            call_id: None,
            name: None,
            arguments: None,
        });
    }

    // Add function_call items for each tool call
    for (i, tc) in ir.tool_calls.iter().enumerate() {
        output.push(ResponsesOutputItem {
            id: format!("fc_{}", uuid::Uuid::new_v4()),
            kind: "function_call",
            status: "completed".into(),
            role: None,
            content: vec![],
            call_id: Some(tc.id.clone()),
            name: Some(tc.name.clone()),
            arguments: Some(tc.arguments.to_string()),
        });
        let _ = i; // suppress unused warning
    }

    // Add message item
    output.push(ResponsesOutputItem {
        id: msg_id,
        kind: "message",
        status: "completed".into(),
        role: Some("assistant"),
        content: vec![ResponsesContent {
            kind: "output_text",
            text: Some(ir.content),
            summary: None,
        }],
        call_id: None,
        name: None,
        arguments: None,
    });

    ResponsesOut {
        id: format!("resp_{}", uuid::Uuid::new_v4()),
        object: "response",
        created_at: chrono::Utc::now().timestamp(),
        model: model_name.to_string(),
        status: "completed".into(),
        output,
        usage: ResponsesUsage {
            input_tokens: ir.usage.prompt_tokens,
            output_tokens: ir.usage.completion_tokens,
            total_tokens: ir.usage.total_tokens,
        },
    }
}

pub fn build_worker_body(ir: &IntermediateRequest, model: &str) -> serde_json::Value {
    // Build input items in Responses API format.
    // Each item needs a `type` field: "message", "function_call", or "function_call_output".
    // Flatten: assistant messages with tool_calls → message + function_call items
    let mut input: Vec<serde_json::Value> = Vec::new();
    let mut instructions: Vec<String> = Vec::new();
    if let Some(system) = ir.system.as_deref()
        && !system.trim().is_empty()
    {
        instructions.push(system.to_string());
    }
    for m in &ir.messages {
        match m.role {
            Role::System => {
                if !m.content.trim().is_empty()
                    && !instructions.iter().any(|existing| existing == &m.content)
                {
                    instructions.push(m.content.clone());
                }
            }
            Role::Tool => {
                let mut item = serde_json::json!({
                    "type": "function_call_output",
                    "output": m.content,
                });
                if let Some(ref call_id) = m.tool_call_id {
                    item["call_id"] = serde_json::json!(call_id);
                }
                input.push(item);
            }
            Role::Assistant => {
                // Emit the assistant message (if it has text content)
                if !m.content.is_empty() {
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": "assistant",
                        "content": m.content,
                    }));
                }
                // Emit function_call items for each tool call
                for tc in &m.tool_calls {
                    input.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments.to_string(),
                    }));
                }
            }
            Role::User => {
                input.push(serde_json::json!({
                    "type": "message",
                    "role": "user",
                    "content": m.content,
                }));
            }
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "input": input,
    });

    if !instructions.is_empty() {
        body["instructions"] = serde_json::json!(instructions.join("\n\n"));
    }

    if let Some(temp) = ir.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(top_p) = ir.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }

    // Inject tool definitions
    if !ir.tools.is_empty() {
        let tools: Vec<serde_json::Value> = ir
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
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
        let mut reasoning = serde_json::json!({
            "effort": effort_str,
        });
        if thinking.summary {
            reasoning["summary"] = serde_json::json!("auto");
        }
        body["reasoning"] = reasoning;
    }

    // Enable SSE streaming if requested
    if ir.stream {
        body["stream"] = serde_json::json!(true);
    }

    merge_native_object(&mut body, &ir.native, NATIVE_OPENAI_RESPONSES);
    body
}

pub fn parse_worker_response(
    body: &serde_json::Value,
    model: &str,
    duration_ms: u64,
    cost_usd: Option<f64>,
) -> Result<WorkerResult, OpenFusionError> {
    let output_items = body["output"]
        .as_array()
        .ok_or_else(|| OpenFusionError::Protocol("missing output array".into()))?;

    let content = output_items
        .iter()
        .filter_map(|item| {
            item["content"].as_array().map(|contents| {
                contents
                    .iter()
                    .filter_map(|c| c["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Extract tool calls from output items (type: "function_call")
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    // Extract thinking text from output items where type == "reasoning"
    let mut thinking_parts: Vec<&str> = Vec::new();
    for item in output_items {
        if item["type"].as_str() == Some("function_call") {
            tool_calls.push(ToolCall {
                id: item["call_id"].as_str().unwrap_or("").to_string(),
                name: item["name"].as_str().unwrap_or("").to_string(),
                arguments: match item["arguments"].clone() {
                    serde_json::Value::Object(map) => serde_json::Value::Object(map),
                    serde_json::Value::String(s) => serde_json::from_str(&s)
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                    _ => serde_json::Value::Object(Default::default()),
                },
            });
        }
        if item["type"].as_str() == Some("reasoning")
            && let Some(summary) = item["summary"].as_array()
        {
            for s in summary {
                if let Some(text) = s["text"].as_str() {
                    thinking_parts.push(text);
                }
            }
        }
    }
    let thinking = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join(""))
    };

    let native = crate::protocol::native_from_body_excluding(
        NATIVE_OPENAI_RESPONSES,
        body,
        &[
            "id",
            "object",
            "created_at",
            "model",
            "status",
            "output",
            "usage",
        ],
    );

    Ok(WorkerResult {
        model: model.to_string(),
        name: String::new(),
        api: "openai-responses".into(),
        success: true,
        response: Some(IntermediateResponse {
            id: body["id"].as_str().map(String::from),
            content,
            status: body["status"].as_str().map(String::from),
            finish_reason: "stop".into(),
            usage: Usage {
                prompt_tokens: body["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: body["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: body["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
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
    fn responses_request_preserves_codex_function_tools() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "fusion",
            "input": "ask",
            "tools": [
                {
                    "type": "function",
                    "name": "request_user_input",
                    "description": "Ask user",
                    "parameters": {
                        "type": "object",
                        "properties": {"questions": {"type": "array"}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "shell_command",
                        "description": "Run shell",
                        "parameters": {"type": "object"}
                    }
                }
            ]
        }))
        .unwrap();

        let ir = to_intermediate(req).unwrap();
        assert_eq!(ir.tools.len(), 2);
        assert_eq!(ir.tools[0].name, "request_user_input");
        assert_eq!(
            ir.tools[0].parameters["properties"]["questions"]["type"],
            "array"
        );
        assert_eq!(ir.tools[1].name, "shell_command");
    }

    #[test]
    fn responses_request_accepts_non_codex_tool_schema_aliases() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "fusion",
            "input": "ask",
            "tools": [
                {
                    "type": "function",
                    "name": "anthropic_style",
                    "description": "Compatibility fallback",
                    "input_schema": {"type": "object"}
                },
                {
                    "type": "function",
                    "function": {
                        "name": "chat_completions_style",
                        "description": "Compatibility fallback",
                        "parameters": {"type": "object"}
                    }
                }
            ]
        }))
        .unwrap();

        let ir = to_intermediate(req).unwrap();
        assert_eq!(ir.tools.len(), 2);
        assert_eq!(ir.tools[0].name, "anthropic_style");
        assert_eq!(ir.tools[1].name, "chat_completions_style");
    }

    #[test]
    fn responses_request_preserves_reasoning_config() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "fusion",
            "input": "ask",
            "reasoning": {
                "effort": "high",
                "summary": "auto"
            }
        }))
        .unwrap();

        let ir = to_intermediate(req).unwrap();
        let thinking = ir.thinking.expect("reasoning should map to thinking");
        assert_eq!(thinking.effort, ThinkingEffort::High);
        assert!(thinking.summary);
    }

    #[test]
    fn responses_request_maps_codex_reasoning_effort_aliases() {
        for (wire_effort, expected) in [
            ("none", ThinkingEffort::Disabled),
            ("minimal", ThinkingEffort::Low),
            ("low", ThinkingEffort::Low),
            ("medium", ThinkingEffort::Medium),
            ("high", ThinkingEffort::High),
            ("xhigh", ThinkingEffort::Max),
            ("max", ThinkingEffort::Max),
            ("ultra", ThinkingEffort::Max),
        ] {
            let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
                "model": "fusion",
                "input": "ask",
                "reasoning": {
                    "effort": wire_effort,
                    "summary": "none"
                }
            }))
            .unwrap();

            let ir = to_intermediate(req).unwrap();
            let thinking = ir.thinking.expect("reasoning should map to thinking");
            assert_eq!(thinking.effort, expected, "{wire_effort}");
            assert!(!thinking.summary, "{wire_effort}");
        }
    }

    #[test]
    fn responses_worker_body_includes_reasoning_from_ir() {
        let ir = IntermediateRequest {
            messages: vec![Message {
                role: Role::User,
                content: "ask".into(),
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
            thinking: Some(ThinkingConfig {
                effort: ThinkingEffort::Max,
                budget_tokens: None,
                summary: true,
            }),
            session_id: None,
            system: None,
            stream: true,
            metadata: Default::default(),
            native: Default::default(),
        };

        let body = build_worker_body(&ir, "model");
        assert_eq!(body["reasoning"]["effort"], "xhigh");
        assert_eq!(body["reasoning"]["summary"], "auto");
    }

    #[test]
    fn responses_worker_body_moves_system_messages_to_instructions() {
        let ir = IntermediateRequest {
            messages: vec![
                Message {
                    role: Role::System,
                    content: "judge system".into(),
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: vec![],
                },
                Message {
                    role: Role::User,
                    content: "ask".into(),
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: vec![],
                },
            ],
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
            system: Some("client instructions".into()),
            stream: true,
            metadata: Default::default(),
            native: Default::default(),
        };

        let body = build_worker_body(&ir, "model");
        let input = body["input"].as_array().expect("input should be an array");

        assert_eq!(body["instructions"], "client instructions\n\njudge system");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
        assert!(
            input
                .iter()
                .all(|item| item["role"].as_str() != Some("developer"))
        );
    }

    #[test]
    fn responses_worker_body_omits_max_output_tokens_for_codex_compatibility() {
        let ir = IntermediateRequest {
            messages: vec![Message {
                role: Role::User,
                content: "ask".into(),
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
            stream: true,
            metadata: Default::default(),
            native: Default::default(),
        };

        let body = build_worker_body(&ir, "model");

        assert!(body.get("max_output_tokens").is_none());
    }
}
