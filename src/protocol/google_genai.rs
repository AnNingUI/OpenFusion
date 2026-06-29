//! Google Generative AI 鈫?Intermediate format.
//!
//! POST /v1beta/models/{model}:generateContent
//! Request:  { contents[{role,parts[{text}]}], systemInstruction?, generationConfig? }
//! Response: { candidates[{content{parts[{text}],role}, finishReason}], usageMetadata }

use super::gemini_shadow::{GeminiAssistantTurn, GeminiToolCallMeta};
use crate::error::OpenFusionError;
use crate::protocol::{
    IntermediateRequest, IntermediateResponse, Message, NATIVE_GEMINI, Role, ThinkingConfig,
    ThinkingEffort, ToolCall, ToolDef, Usage, WorkerResult, merge_native_object, native_from_extra,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;

/// Prefix used for tool call ids synthesized when Gemini's `functionCall`
/// omits `id`. These ids are client/proxy-visible only and must not be sent
/// upstream to Gemini as `functionCall.id` or `functionResponse.id`.
pub(crate) const SYNTHESIZED_ID_PREFIX: &str = "gemini_synth_";

pub(crate) fn synthesize_tool_call_id() -> String {
    format!("{SYNTHESIZED_ID_PREFIX}{}", uuid::Uuid::new_v4().simple())
}

pub(crate) fn is_synthesized_tool_call_id(id: &str) -> bool {
    id.starts_with(SYNTHESIZED_ID_PREFIX)
}

#[derive(Debug, Deserialize)]
pub struct GenAiRequest {
    pub contents: Vec<GenAiContent>,
    #[serde(default)]
    #[serde(alias = "systemInstruction")]
    pub system_instruction: Option<GenAiSystemInstruction>,
    #[serde(default)]
    #[serde(alias = "generationConfig")]
    pub generation_config: Option<GenAiConfig>,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Request streaming SSE response
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<GenAiTool>,
    #[serde(default)]
    #[serde(alias = "toolConfig")]
    pub tool_config: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiTool {
    #[serde(
        default,
        rename = "functionDeclarations",
        alias = "function_declarations"
    )]
    pub function_declarations: Vec<GenAiFunctionDeclaration>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiFunctionDeclaration {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct GenAiContent {
    pub role: Option<String>,
    pub parts: Vec<GenAiPart>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiPart {
    pub text: Option<String>,
    #[serde(rename = "functionCall")]
    pub function_call: Option<GenAiFunctionCall>,
    #[serde(rename = "functionResponse")]
    pub function_response: Option<GenAiFunctionResponse>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiFunctionCall {
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct GenAiFunctionResponse {
    #[serde(default)]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub response: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct GenAiSystemInstruction {
    pub parts: Vec<GenAiPart>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiConfig {
    #[serde(alias = "maxOutputTokens")]
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    #[serde(default)]
    #[serde(alias = "topP")]
    pub top_p: Option<f32>,
    #[serde(default)]
    #[serde(alias = "stopSequences")]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    #[serde(alias = "thinkingConfig")]
    pub thinking_config: Option<GenAiThinkingConfig>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiThinkingConfig {
    #[serde(default)]
    #[serde(alias = "includeThoughts")]
    pub include_thoughts: Option<bool>,
    #[serde(default)]
    #[serde(alias = "thinkingBudget")]
    pub thinking_budget: Option<i32>,
    #[serde(default)]
    #[serde(alias = "thinkingLevel")]
    pub thinking_level: Option<String>,
}

fn default_max_output_tokens() -> u32 {
    2048
}

fn parse_gemini_thinking_config(config: Option<&GenAiConfig>) -> Option<ThinkingConfig> {
    let thinking_config = config?.thinking_config.as_ref()?;
    let budget = thinking_config.thinking_budget;
    let effort = if let Some(level) = thinking_config.thinking_level.as_deref() {
        match level {
            "LOW" | "low" => ThinkingEffort::Low,
            "MEDIUM" | "medium" => ThinkingEffort::Medium,
            "HIGH" | "high" => ThinkingEffort::High,
            _ => ThinkingEffort::Adaptive,
        }
    } else {
        match budget {
            Some(0) => ThinkingEffort::Disabled,
            Some(n) if n > 0 => ThinkingEffort::High,
            _ if thinking_config.include_thoughts == Some(false) => ThinkingEffort::Disabled,
            _ => ThinkingEffort::Adaptive,
        }
    };
    Some(ThinkingConfig {
        effort,
        budget_tokens: budget,
        summary: false,
    })
}

fn map_tool_choice_to_gemini(tool_choice: &serde_json::Value) -> serde_json::Value {
    match tool_choice {
        serde_json::Value::String(choice) => match choice.as_str() {
            "none" => json!({"functionCallingConfig": {"mode": "NONE"}}),
            "required" | "any" => json!({"functionCallingConfig": {"mode": "ANY"}}),
            _ => json!({"functionCallingConfig": {"mode": "AUTO"}}),
        },
        serde_json::Value::Object(object) => {
            if object.contains_key("functionCallingConfig") {
                return tool_choice.clone();
            }
            let choice_type = object
                .get("type")
                .and_then(|value| value.as_str())
                .unwrap_or("auto");
            match choice_type {
                "none" => json!({"functionCallingConfig": {"mode": "NONE"}}),
                "any" | "required" => json!({"functionCallingConfig": {"mode": "ANY"}}),
                "tool" | "function" => {
                    let name = object
                        .get("name")
                        .or_else(|| object.get("function").and_then(|f| f.get("name")))
                        .and_then(|value| value.as_str())
                        .unwrap_or("");
                    json!({
                        "functionCallingConfig": {
                            "mode": "ANY",
                            "allowedFunctionNames": [name]
                        }
                    })
                }
                _ => json!({"functionCallingConfig": {"mode": "AUTO"}}),
            }
        }
        _ => json!({"functionCallingConfig": {"mode": "AUTO"}}),
    }
}

fn map_gemini_tool_config_to_common(tool_config: serde_json::Value) -> serde_json::Value {
    let Some(config) = tool_config
        .get("functionCallingConfig")
        .or_else(|| tool_config.get("function_calling_config"))
    else {
        return serde_json::json!("auto");
    };
    let mode = config
        .get("mode")
        .and_then(|value| value.as_str())
        .unwrap_or("AUTO");
    match mode {
        "NONE" => serde_json::json!("none"),
        "ANY" => {
            if let Some(name) = config
                .get("allowedFunctionNames")
                .or_else(|| config.get("allowed_function_names"))
                .and_then(|value| value.as_array())
                .and_then(|names| names.first())
                .and_then(|name| name.as_str())
            {
                serde_json::json!({"type": "function", "function": {"name": name}})
            } else {
                serde_json::json!("required")
            }
        }
        _ => serde_json::json!("auto"),
    }
}

fn gemini_uses_thinking_level(model: &str) -> bool {
    model.starts_with("gemini-3")
}

fn thinking_effort_to_gemini_level(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Disabled | ThinkingEffort::Low => "LOW",
        ThinkingEffort::Medium | ThinkingEffort::Adaptive => "MEDIUM",
        ThinkingEffort::High | ThinkingEffort::Max => "HIGH",
    }
}

// 鈹€鈹€ Conversion 鈹€鈹€

pub fn to_intermediate(
    req: GenAiRequest,
    _model: &str,
) -> Result<IntermediateRequest, OpenFusionError> {
    let mut messages = Vec::new();

    // Validate
    if req.contents.is_empty() {
        return Err(OpenFusionError::Protocol(
            "contents must not be empty".into(),
        ));
    }
    if let Some(ref cfg) = req.generation_config {
        if cfg.max_output_tokens == 0 {
            return Err(OpenFusionError::Protocol(
                "max_output_tokens must be > 0".into(),
            ));
        }
        if let Some(temp) = cfg.temperature
            && (!(0.0..=2.0).contains(&temp))
        {
            return Err(OpenFusionError::Protocol(format!(
                "temperature must be 0.0..=2.0, got {temp}"
            )));
        }
    }

    let mut system_for_ir = None;
    if let Some(sys) = req.system_instruction {
        let sys_text = sys
            .parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join("\n");
        if !sys_text.is_empty() {
            system_for_ir = Some(sys_text.clone());
            messages.push(Message {
                role: Role::System,
                content: sys_text,
                tool_call_id: None,
                tool_is_error: None,
                tool_calls: vec![],
            });
        }
    }

    for content in req.contents {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_results = Vec::new();
        for part in content.parts {
            if let Some(text) = part.text {
                text_parts.push(text);
            }
            if let Some(fc) = part.function_call {
                let id = fc
                    .id
                    .unwrap_or_else(|| format!("{}_{}", fc.name, tool_calls.len()));
                tool_calls.push(ToolCall {
                    id,
                    name: fc.name,
                    arguments: fc.args,
                });
            }
            if let Some(fr) = part.function_response {
                let content = fr
                    .response
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| fr.response.to_string());
                tool_results.push(Message {
                    role: Role::Tool,
                    content,
                    tool_call_id: Some(fr.id.unwrap_or(fr.name)),
                    tool_is_error: None,
                    tool_calls: vec![],
                });
            }
        }
        let text = text_parts.join("\n");
        let role = match content.role.as_deref() {
            Some("model") => Role::Assistant,
            Some("user") | None => Role::User,
            _ => Role::User,
        };
        if !tool_results.is_empty() {
            messages.extend(tool_results);
        } else {
            messages.push(Message {
                role,
                content: text,
                tool_call_id: None,
                tool_is_error: None,
                tool_calls,
            });
        }
    }

    let thinking = parse_gemini_thinking_config(req.generation_config.as_ref());
    let (max_tokens, temperature, top_p, stop) = match req.generation_config {
        Some(cfg) => (
            cfg.max_output_tokens,
            cfg.temperature,
            cfg.top_p,
            cfg.stop_sequences,
        ),
        None => (2048, None, None, vec![]),
    };

    let tools: Vec<ToolDef> = req
        .tools
        .into_iter()
        .flat_map(|tool| tool.function_declarations)
        .map(|decl| ToolDef {
            name: decl.name,
            description: decl.description,
            parameters: decl.parameters,
        })
        .collect();
    let tool_choice = req.tool_config.map(map_gemini_tool_config_to_common);
    let native = native_from_extra(NATIVE_GEMINI, req.extra);

    Ok(IntermediateRequest {
        messages,
        max_tokens,
        temperature,
        top_p,
        stop,
        api_key: req.api_key,
        tools,
        tool_choice,
        parallel_tool_calls: None,
        thinking,
        session_id: None,
        system: system_for_ir,
        stream: req.stream,
        metadata: Default::default(),
        native,
    })
}

pub fn from_intermediate(ir: IntermediateResponse) -> serde_json::Value {
    let mut parts = Vec::new();

    // Add thinking part if present
    if let Some(ref thinking) = ir.thinking
        && !thinking.is_empty()
    {
        parts.push(serde_json::json!({
            "text": thinking,
            "thought": true
        }));
    }

    // Add text part
    if !ir.content.is_empty() {
        parts.push(serde_json::json!({"text": ir.content}));
    }

    for tc in &ir.tool_calls {
        let mut function_call = serde_json::json!({
            "name": tc.name,
            "args": tc.arguments,
        });
        if !tc.id.is_empty() && !is_synthesized_tool_call_id(&tc.id) {
            function_call["id"] = serde_json::json!(tc.id);
        }
        parts.push(serde_json::json!({ "functionCall": function_call }));
    }

    serde_json::json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": parts
            },
            "finishReason": ir.finish_reason
        }],
        "usageMetadata": {
            "promptTokenCount": ir.usage.prompt_tokens,
            "candidatesTokenCount": ir.usage.completion_tokens,
            "totalTokenCount": ir.usage.total_tokens
        }
    })
}

#[allow(dead_code)]
pub fn build_worker_body(ir: &IntermediateRequest, model: &str) -> serde_json::Value {
    build_worker_body_with_shadow(ir, model, &[])
}

pub fn build_worker_body_with_shadow(
    ir: &IntermediateRequest,
    model: &str,
    shadow_turns: &[GeminiAssistantTurn],
) -> serde_json::Value {
    let mut parts: Vec<serde_json::Value> = Vec::new();
    let mut sys_parts: Vec<serde_json::Value> = Vec::new();
    let mut tool_name_by_id = build_tool_name_map_from_shadow_turns(shadow_turns);
    let thought_signature_by_id = build_thought_signature_map_from_shadow_turns(shadow_turns);

    for m in &ir.messages {
        match m.role {
            Role::System => {
                sys_parts.push(serde_json::json!({"text": m.content}));
            }
            Role::User => {
                parts.push(serde_json::json!({
                    "role": "user",
                    "parts": [{"text": m.content}],
                }));
            }
            Role::Assistant if !m.tool_calls.is_empty() => {
                let p = find_matching_shadow_turn_for_message(m, shadow_turns)
                    .and_then(|turn| shadow_parts(&turn.assistant_content))
                    .unwrap_or_else(|| {
                        let mut p: Vec<serde_json::Value> = Vec::new();
                        if !m.content.is_empty() {
                            p.push(serde_json::json!({"text": m.content}));
                        }
                        for tc in &m.tool_calls {
                            tool_name_by_id.insert(tc.id.clone(), tc.name.clone());
                            let mut function_call = serde_json::json!({
                                "name": tc.name,
                                "args": tc.arguments,
                            });
                            if !tc.id.is_empty() && !is_synthesized_tool_call_id(&tc.id) {
                                function_call["id"] = serde_json::json!(tc.id);
                            }
                            if let Some(sig) = thought_signature_by_id.get(&tc.id) {
                                function_call["thoughtSignature"] = serde_json::json!(sig);
                            }
                            p.push(serde_json::json!({ "functionCall": function_call }));
                        }
                        p
                    });
                merge_tool_names_from_parts(&p, &mut tool_name_by_id);
                parts.push(serde_json::json!({
                    "role": "model",
                    "parts": p,
                }));
            }
            Role::Assistant => {
                parts.push(serde_json::json!({
                    "role": "model",
                    "parts": [{"text": m.content}],
                }));
            }
            Role::Tool => {
                let id = m.tool_call_id.as_deref();
                let name = id
                    .and_then(|id| tool_name_by_id.get(id).map(String::as_str))
                    .unwrap_or_else(|| id.unwrap_or("unknown"));
                let mut function_response = serde_json::json!({
                    "name": name,
                    "response": { "content": m.content },
                });
                if let Some(id) = id
                    && id != name
                    && !is_synthesized_tool_call_id(id)
                {
                    function_response["id"] = serde_json::json!(id);
                }
                parts.push(serde_json::json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": function_response
                    }],
                }));
            }
        }
    }

    let mut gen_config = serde_json::json!({
        "maxOutputTokens": ir.max_tokens,
    });
    if let Some(temp) = ir.temperature {
        gen_config["temperature"] = serde_json::json!(temp);
    }
    if let Some(top_p) = ir.top_p {
        gen_config["topP"] = serde_json::json!(top_p);
    }
    if !ir.stop.is_empty() {
        gen_config["stopSequences"] = serde_json::json!(ir.stop);
    }

    let mut body = serde_json::json!({
        "contents": parts,
        "generationConfig": gen_config,
    });

    if !sys_parts.is_empty() {
        body["systemInstruction"] = serde_json::json!({
            "parts": sys_parts,
        });
    }

    // Inject thinking configuration into generationConfig
    if let Some(ref thinking) = ir.thinking {
        body["generationConfig"]["thinkingConfig"] = if gemini_uses_thinking_level(model) {
            serde_json::json!({
                "includeThoughts": thinking.effort != ThinkingEffort::Disabled,
                "thinkingLevel": thinking_effort_to_gemini_level(thinking.effort),
            })
        } else {
            let budget = if thinking.effort == ThinkingEffort::Disabled {
                0
            } else {
                thinking.budget_tokens.unwrap_or(-1)
            };
            serde_json::json!({
                "includeThoughts": thinking.effort != ThinkingEffort::Disabled,
                "thinkingBudget": budget,
            })
        };
    }

    // Inject tool definitions as functionDeclarations
    if !ir.tools.is_empty() {
        let function_declarations: Vec<serde_json::Value> = ir
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();
        body["tools"] = serde_json::json!([{
            "functionDeclarations": function_declarations,
        }]);
    }
    if let Some(ref tool_choice) = ir.tool_choice {
        body["toolConfig"] = map_tool_choice_to_gemini(tool_choice);
    }

    // Enable SSE streaming if requested
    if ir.stream {
        body["stream"] = serde_json::json!(true);
    }

    merge_native_object(&mut body, &ir.native, NATIVE_GEMINI);
    body
}

pub fn parse_worker_response(
    body: &serde_json::Value,
    model: &str,
    duration_ms: u64,
    cost_usd: Option<f64>,
) -> Result<WorkerResult, OpenFusionError> {
    let candidate = body["candidates"]
        .as_array()
        .and_then(|c| c.first())
        .ok_or_else(|| OpenFusionError::Protocol("missing candidates[0]".into()))?;

    let content = candidate["content"]["parts"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    // Extract thinking text from parts where thought == true
    let thinking = candidate["content"]["parts"]
        .as_array()
        .map(|parts| {
            let thought_parts: Vec<&str> = parts
                .iter()
                .filter(|p| p["thought"].as_bool() == Some(true))
                .filter_map(|p| p["text"].as_str())
                .collect();
            if thought_parts.is_empty() {
                None
            } else {
                Some(thought_parts.join(""))
            }
        })
        .unwrap_or(None);

    let finish_reason = candidate["finishReason"]
        .as_str()
        .unwrap_or("STOP")
        .to_string();

    // Extract function calls from candidate parts.
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    if let Some(parts) = candidate["content"]["parts"].as_array() {
        for part in parts {
            if let Some(fc) = part.get("functionCall") {
                let name = fc["name"].as_str().unwrap_or("").to_string();
                let id = fc["id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(synthesize_tool_call_id);
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: fc["args"].clone(),
                });
            }
        }
    }

    let native = crate::protocol::native_from_body_excluding(
        NATIVE_GEMINI,
        body,
        &["responseId", "candidates", "usageMetadata", "modelVersion"],
    );

    Ok(WorkerResult {
        model: model.to_string(),
        name: String::new(),
        api: "google-generative-ai".into(),
        success: true,
        response: Some(IntermediateResponse {
            id: body["responseId"].as_str().map(String::from),
            content,
            status: None,
            finish_reason,
            usage: Usage {
                prompt_tokens: body["usageMetadata"]["promptTokenCount"]
                    .as_u64()
                    .unwrap_or(0) as u32,
                completion_tokens: body["usageMetadata"]["candidatesTokenCount"]
                    .as_u64()
                    .unwrap_or(0) as u32,
                total_tokens: body["usageMetadata"]["totalTokenCount"]
                    .as_u64()
                    .unwrap_or(0) as u32,
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

pub fn rectify_response_missing_function_call_ids(body: &mut serde_json::Value) {
    let Some(parts) = body["candidates"]
        .as_array_mut()
        .and_then(|candidates| candidates.first_mut())
        .and_then(|candidate| candidate.get_mut("content"))
        .and_then(|content| content.get_mut("parts"))
        .and_then(|parts| parts.as_array_mut())
    else {
        return;
    };
    rectify_missing_function_call_ids(parts);
}

pub fn extract_assistant_shadow_turn(body: &serde_json::Value) -> Option<GeminiAssistantTurn> {
    let content = body["candidates"].as_array()?.first()?.get("content")?;
    let shadow_content = content.clone();
    let parts = shadow_content.get("parts")?.as_array()?;
    Some(GeminiAssistantTurn::new(
        shadow_content.clone(),
        extract_tool_call_meta(parts),
    ))
}

fn rectify_missing_function_call_ids(parts: &mut [serde_json::Value]) {
    for part in parts {
        let Some(function_call) = part
            .get_mut("functionCall")
            .and_then(|value| value.as_object_mut())
        else {
            continue;
        };
        let needs_synth = function_call
            .get("id")
            .and_then(|value| value.as_str())
            .map(|id| id.is_empty())
            .unwrap_or(true);
        if needs_synth {
            function_call.insert("id".to_string(), json!(synthesize_tool_call_id()));
        }
    }
}

fn shadow_parts(content: &serde_json::Value) -> Option<Vec<serde_json::Value>> {
    let mut parts = content
        .get("parts")
        .and_then(|value| value.as_array())
        .cloned()
        .or_else(|| content.as_array().cloned())?;
    for part in &mut parts {
        let Some(function_call) = part
            .get_mut("functionCall")
            .and_then(|value| value.as_object_mut())
        else {
            continue;
        };
        let drop_id = function_call
            .get("id")
            .and_then(|value| value.as_str())
            .map(|id| id.is_empty() || is_synthesized_tool_call_id(id))
            .unwrap_or(true);
        if drop_id {
            function_call.remove("id");
        }
    }
    Some(parts)
}

fn extract_tool_call_meta(parts: &[serde_json::Value]) -> Vec<GeminiToolCallMeta> {
    parts
        .iter()
        .filter_map(|part| {
            let function_call = part.get("functionCall")?;
            let id = function_call
                .get("id")
                .and_then(|value| value.as_str())
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(synthesize_tool_call_id);
            Some(GeminiToolCallMeta::new(
                Some(id),
                function_call
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or(""),
                function_call
                    .get("args")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                part.get("thoughtSignature")
                    .or_else(|| part.get("thought_signature"))
                    .and_then(|value| value.as_str()),
            ))
        })
        .collect()
}

fn merge_tool_names_from_shadow(
    turn: &GeminiAssistantTurn,
    tool_name_by_id: &mut HashMap<String, String>,
) {
    for tool_call in &turn.tool_calls {
        if let Some(id) = &tool_call.id {
            tool_name_by_id.insert(id.clone(), tool_call.name.clone());
        }
    }

    if let Some(parts) = shadow_parts(&turn.assistant_content) {
        merge_tool_names_from_parts(&parts, tool_name_by_id);
    }
}

fn build_tool_name_map_from_shadow_turns(
    shadow_turns: &[GeminiAssistantTurn],
) -> HashMap<String, String> {
    let mut tool_name_by_id = HashMap::new();
    for turn in shadow_turns {
        merge_tool_names_from_shadow(turn, &mut tool_name_by_id);
    }
    tool_name_by_id
}

fn build_thought_signature_map_from_shadow_turns(
    shadow_turns: &[GeminiAssistantTurn],
) -> HashMap<String, String> {
    let mut thought_signature_by_id = HashMap::new();
    for turn in shadow_turns {
        merge_thought_signatures_from_shadow(turn, &mut thought_signature_by_id);
    }
    thought_signature_by_id
}

fn merge_thought_signatures_from_shadow(
    turn: &GeminiAssistantTurn,
    thought_signature_by_id: &mut HashMap<String, String>,
) {
    for tool_call in &turn.tool_calls {
        if let (Some(id), Some(sig)) = (&tool_call.id, &tool_call.thought_signature) {
            thought_signature_by_id.insert(id.clone(), sig.clone());
        }
    }
}

fn merge_tool_names_from_parts(
    parts: &[serde_json::Value],
    tool_name_by_id: &mut HashMap<String, String>,
) {
    for part in parts {
        let Some(function_call) = part.get("functionCall") else {
            continue;
        };
        let Some(id) = function_call.get("id").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(name) = function_call.get("name").and_then(|value| value.as_str()) else {
            continue;
        };
        if !id.is_empty() && !name.is_empty() {
            tool_name_by_id.insert(id.to_string(), name.to_string());
        }
    }
}

fn find_matching_shadow_turn_for_message<'a>(
    message: &Message,
    shadow_turns: &'a [GeminiAssistantTurn],
) -> Option<&'a GeminiAssistantTurn> {
    if message.tool_calls.is_empty() {
        return None;
    }

    shadow_turns.iter().rev().find(|turn| {
        turn.tool_calls.iter().any(|shadow_call| {
            message.tool_calls.iter().any(|message_call| {
                shadow_call.id.as_deref() == Some(message_call.id.as_str())
                    || shadow_call.name == message_call.name
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_preserves_function_call_id_and_response_name() {
        let body = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"id": "call_1", "name": "inspect_input", "args": {"path": "sample-target"}}
                }]},
                "finishReason": "STOP"
            }]
        });

        let result = parse_worker_response(&body, "gemini", 1, None).unwrap();
        let response = result.response.unwrap();
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].name, "inspect_input");

        let ir = IntermediateRequest {
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: String::new(),
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: response.tool_calls,
                },
                Message {
                    role: Role::Tool,
                    content: "ok".into(),
                    tool_call_id: Some("call_1".into()),
                    tool_is_error: None,
                    tool_calls: vec![],
                },
            ],
            max_tokens: 100,
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
        let request = build_worker_body(&ir, "gemini");
        assert_eq!(
            request["contents"][1]["parts"][0]["functionResponse"]["name"],
            "inspect_input"
        );
        assert_eq!(
            request["contents"][1]["parts"][0]["functionResponse"]["id"],
            "call_1"
        );
    }

    #[test]
    fn genai_request_preserves_thinking_config() {
        let req: GenAiRequest = serde_json::from_value(serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "think"}]}],
            "generationConfig": {
                "maxOutputTokens": 128,
                "thinkingConfig": {
                    "includeThoughts": true,
                    "thinkingBudget": 256
                }
            }
        }))
        .unwrap();

        let ir = to_intermediate(req, "gemini").unwrap();
        let thinking = ir.thinking.expect("thinkingConfig should map to thinking");
        assert_eq!(thinking.effort, ThinkingEffort::High);
        assert_eq!(thinking.budget_tokens, Some(256));
    }

    #[test]
    fn genai_request_maps_zero_thinking_budget_to_disabled() {
        let req: GenAiRequest = serde_json::from_value(serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "think"}]}],
            "generationConfig": {
                "thinkingConfig": {
                    "includeThoughts": false,
                    "thinkingBudget": 0
                }
            }
        }))
        .unwrap();

        let ir = to_intermediate(req, "gemini").unwrap();
        let thinking = ir.thinking.expect("thinkingConfig should map to thinking");
        assert_eq!(thinking.effort, ThinkingEffort::Disabled);
        assert_eq!(thinking.budget_tokens, Some(0));
    }

    #[test]
    fn genai_request_preserves_thinking_level() {
        let req: GenAiRequest = serde_json::from_value(serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "think"}]}],
            "generationConfig": {
                "thinkingConfig": {
                    "includeThoughts": true,
                    "thinkingLevel": "HIGH"
                }
            }
        }))
        .unwrap();

        let ir = to_intermediate(req, "gemini-3-pro-preview").unwrap();
        let thinking = ir.thinking.expect("thinkingLevel should map to thinking");
        assert_eq!(thinking.effort, ThinkingEffort::High);
        assert_eq!(thinking.budget_tokens, None);
    }

    #[test]
    fn gemini3_worker_body_uses_thinking_level_not_budget() {
        let ir = IntermediateRequest {
            messages: vec![Message {
                role: Role::User,
                content: "think".into(),
                tool_call_id: None,
                tool_is_error: None,
                tool_calls: vec![],
            }],
            max_tokens: 100,
            temperature: None,
            top_p: None,
            stop: vec![],
            api_key: None,
            tools: vec![],
            tool_choice: None,
            parallel_tool_calls: None,
            thinking: Some(ThinkingConfig {
                effort: ThinkingEffort::High,
                budget_tokens: Some(8192),
                summary: false,
            }),
            session_id: None,
            system: None,
            stream: false,
            metadata: Default::default(),
            native: Default::default(),
        };

        let body = build_worker_body(&ir, "gemini-3-pro-preview");
        let thinking_config = &body["generationConfig"]["thinkingConfig"];
        assert_eq!(thinking_config["includeThoughts"], true);
        assert_eq!(thinking_config["thinkingLevel"], "HIGH");
        assert!(thinking_config.get("thinkingBudget").is_none());
    }

    #[test]
    fn gemini_synthesized_id_is_shadowed_and_not_replayed_upstream() {
        let mut body = serde_json::json!({
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {
                        "name": "inspect_input",
                        "args": {"path": "sample-target"},
                        "thoughtSignature": "sig-tool-1"
                    }
                }]},
                "finishReason": "STOP"
            }]
        });

        rectify_response_missing_function_call_ids(&mut body);
        let result = parse_worker_response(&body, "gemini", 1, None).unwrap();
        let response = result.response.unwrap();
        let synthesized_id = response.tool_calls[0].id.clone();
        assert!(is_synthesized_tool_call_id(&synthesized_id));

        let shadow_turn = extract_assistant_shadow_turn(&body).unwrap();
        let ir = IntermediateRequest {
            messages: vec![
                Message {
                    role: Role::Assistant,
                    content: String::new(),
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: response.tool_calls,
                },
                Message {
                    role: Role::Tool,
                    content: "ok".into(),
                    tool_call_id: Some(synthesized_id),
                    tool_is_error: None,
                    tool_calls: vec![],
                },
            ],
            max_tokens: 100,
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
        let request = build_worker_body_with_shadow(&ir, "gemini", &[shadow_turn]);

        let function_call = &request["contents"][0]["parts"][0]["functionCall"];
        assert_eq!(function_call["name"], "inspect_input");
        assert!(function_call.get("id").is_none());
        assert_eq!(function_call["thoughtSignature"], "sig-tool-1");

        let function_response = &request["contents"][1]["parts"][0]["functionResponse"];
        assert_eq!(function_response["name"], "inspect_input");
        assert!(function_response.get("id").is_none());
        assert_eq!(function_response["response"]["content"], "ok");
    }

    #[test]
    fn gemini_shadow_store_replays_synthesized_tool_turn_by_session() {
        let store = crate::protocol::gemini_shadow::GeminiShadowStore::with_limits(8, 4);
        let synth = synthesize_tool_call_id();
        store.record_assistant_turn(
            "google-generative-ai:gemini",
            "session-1",
            serde_json::json!({
                "parts": [{
                    "functionCall": {
                        "id": synth,
                        "name": "inspect_input",
                        "args": {"path": "src/lib.rs"}
                    }
                }]
            }),
            vec![GeminiToolCallMeta::new(
                Some(synth.clone()),
                "inspect_input",
                serde_json::json!({"path": "src/lib.rs"}),
                None::<String>,
            )],
        );

        let snapshot = store
            .get_session("google-generative-ai:gemini", "session-1")
            .unwrap();
        let ir = IntermediateRequest {
            messages: vec![Message {
                role: Role::Tool,
                content: "ok".into(),
                tool_call_id: Some(synth),
                tool_is_error: None,
                tool_calls: vec![],
            }],
            max_tokens: 100,
            temperature: None,
            top_p: None,
            stop: vec![],
            api_key: None,
            tools: vec![],
            tool_choice: None,
            parallel_tool_calls: None,
            thinking: None,
            session_id: Some("session-1".into()),
            system: None,
            stream: false,
            metadata: Default::default(),
            native: Default::default(),
        };

        let request = build_worker_body_with_shadow(&ir, "gemini", &snapshot.turns);
        let function_response = &request["contents"][0]["parts"][0]["functionResponse"];
        assert_eq!(function_response["name"], "inspect_input");
        assert!(function_response.get("id").is_none());
    }
}
