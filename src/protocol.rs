//! Unified intermediate format for all protocol conversions.
//!
//! Every inbound protocol adapter converts its native request into
//! `IntermediateRequest` and converts `IntermediateResponse` back to
//! its native response format. This isolates protocol concerns from
//! the fusion engine.

pub mod anthropic_messages;
pub mod google_genai;
pub mod openai_completions;
pub mod openai_responses;

use serde::{Deserialize, Serialize};

// ── Intermediate Request ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateRequest {
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub stop: Vec<String>,
    /// Request-level API key override (highest priority)
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

// ── Intermediate Response ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateResponse {
    pub content: String,
    pub finish_reason: String,
    pub usage: Usage,
    pub cost_usd: f64,
    pub duration_ms: u64,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Worker result ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResult {
    pub model: String,
    pub api: String,
    pub success: bool,
    pub response: Option<IntermediateResponse>,
    pub error: Option<String>,
}
