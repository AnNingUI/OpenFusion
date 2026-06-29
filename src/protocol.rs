//! Unified intermediate format for all protocol conversions.
//!
//! Every inbound protocol adapter converts its native request into
//! `IntermediateRequest` and converts `IntermediateResponse` back to
//! its native response format. This isolates protocol concerns from
//! the fusion engine.

#![allow(dead_code)]

pub mod anthropic_messages;
pub mod gemini_shadow;
pub mod google_genai;
pub mod openai_completions;
pub mod openai_responses;
#[cfg(test)]
pub mod stream;

#[cfg(test)]
mod matrix_tests;

use serde::{Deserialize, Serialize};

pub(crate) const NATIVE_OPENAI_CHAT: &str = "openai_chat";
pub(crate) const NATIVE_OPENAI_RESPONSES: &str = "openai_responses";
pub(crate) const NATIVE_ANTHROPIC: &str = "anthropic";
pub(crate) const NATIVE_GEMINI: &str = "gemini";

pub(crate) fn native_from_extra(
    protocol: &str,
    extra: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut native = serde_json::Map::new();
    if !extra.is_empty() {
        native.insert(protocol.to_string(), serde_json::Value::Object(extra));
    }
    native
}

pub(crate) fn native_from_body_excluding(
    protocol: &str,
    body: &serde_json::Value,
    known_keys: &[&str],
) -> serde_json::Map<String, serde_json::Value> {
    let mut native = serde_json::Map::new();
    let Some(object) = body.as_object() else {
        return native;
    };

    let known: std::collections::HashSet<&str> = known_keys.iter().copied().collect();
    let extra: serde_json::Map<String, serde_json::Value> = object
        .iter()
        .filter(|(key, _)| !known.contains(key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();

    if !extra.is_empty() {
        native.insert(protocol.to_string(), serde_json::Value::Object(extra));
    }
    native
}

pub(crate) fn merge_native_object(
    body: &mut serde_json::Value,
    native: &serde_json::Map<String, serde_json::Value>,
    protocol: &str,
) {
    let Some(target) = body.as_object_mut() else {
        return;
    };
    let Some(extra) = native.get(protocol).and_then(|value| value.as_object()) else {
        return;
    };

    for (key, value) in extra {
        target.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

// ── Thinking configuration ──

/// Unified thinking configuration for all protocols.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThinkingConfig {
    pub effort: ThinkingEffort,
    /// Google GenAI specific: token budget (-1 = auto, 0 = disabled)
    pub budget_tokens: Option<i32>,
    /// OpenAI Responses specific: include reasoning summary
    pub summary: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingEffort {
    Disabled,
    Low,
    Medium,
    #[default]
    High,
    Max,
    Adaptive,
}

// ── Tool definitions ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

// ── Intermediate Request ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateRequest {
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Vec<String>,
    /// Request-level API key override (highest priority)
    pub api_key: Option<String>,
    /// Tool definitions to inject into the request
    pub tools: Vec<ToolDef>,
    /// Native tool choice shape from the caller protocol, normalized only when
    /// a target protocol has an equivalent representation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Unified thinking/reasoning configuration
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    /// Conversation/session id used by provider-specific replay stores.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// System prompt �?mapped to each protocol's native field:
    /// Anthropic: `system`, OpenAI Responses: `instructions`,
    /// Gemini: `systemInstruction`, OpenAI Chat: messages[{role:"system"}]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Whether to request streaming SSE from upstream
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub native: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Tool call ID �?only set for Role::Tool messages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Whether a tool result represents a failed tool execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_is_error: Option<bool>,
    /// Tool calls made by the assistant in this message.
    /// Preserved for round-trip fidelity (e.g., Anthropic tool_use blocks).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

// ── Intermediate Response ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    pub finish_reason: String,
    pub usage: Usage,
    pub cost_usd: f64,
    pub duration_ms: u64,
    pub model: String,
    /// Tool calls returned by the model (empty if no tools were called)
    pub tool_calls: Vec<ToolCall>,
    /// Extracted thinking/reasoning text (if present in the response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub native: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub reasoning_tokens: u32,
}

// ── Worker result ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResult {
    pub model: String,
    pub name: String,
    pub api: String,
    pub success: bool,
    pub response: Option<IntermediateResponse>,
    pub error: Option<String>,
}

// ── Streaming events ──

/// Events yielded during streaming execution.
#[cfg(test)]
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A text delta from the model
    Delta(String),
    /// A thinking/reasoning delta
    ThinkingDelta(String),
    /// A tool call delta (index for multi-block ordering)
    ToolCallDelta {
        index: u32,
        id: String,
        name: String,
        arguments: String,
    },
    /// Stream completed with full result
    Done(IntermediateResponse),
}
