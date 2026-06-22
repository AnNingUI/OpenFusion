//! Google Generative AI ↔ Intermediate format.
//!
//! POST /v1beta/models/{model}:generateContent
//! Request:  { contents[{role,parts[{text}]}], systemInstruction?, generationConfig? }
//! Response: { candidates[{content{parts[{text}],role}, finishReason}], usageMetadata }

use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, IntermediateResponse, Message, Role, Usage, WorkerResult};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct GenAiRequest {
    pub contents: Vec<GenAiContent>,
    #[serde(default)]
    pub system_instruction: Option<GenAiSystemInstruction>,
    #[serde(default)]
    pub generation_config: Option<GenAiConfig>,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiContent {
    pub role: Option<String>,
    pub parts: Vec<GenAiPart>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiPart {
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiSystemInstruction {
    pub parts: Vec<GenAiPart>,
}

#[derive(Debug, Deserialize)]
pub struct GenAiConfig {
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: u32,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

fn default_max_output_tokens() -> u32 { 2048 }

// ── Conversion ──

pub fn to_intermediate(req: GenAiRequest, _model: &str) -> Result<IntermediateRequest, OpenFusionError> {
    let mut messages = Vec::new();

    // Validate
    if req.contents.is_empty() {
        return Err(OpenFusionError::Protocol("contents must not be empty".into()));
    }
    if let Some(ref cfg) = req.generation_config {
        if cfg.max_output_tokens == 0 {
            return Err(OpenFusionError::Protocol("max_output_tokens must be > 0".into()));
        }
        if let Some(temp) = cfg.temperature
            && (!(0.0..=2.0).contains(&temp)) {
                return Err(OpenFusionError::Protocol(format!(
                    "temperature must be 0.0..=2.0, got {temp}"
                )));
            }
    }

    if let Some(sys) = req.system_instruction {
        let sys_text = sys.parts.into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join("\n");
        if !sys_text.is_empty() {
            messages.push(Message { role: Role::System, content: sys_text });
        }
    }

    for content in req.contents {
        let text = content.parts.into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join("\n");
        let role = match content.role.as_deref() {
            Some("model") => Role::Assistant,
            Some("user") | None => Role::User,
            _ => Role::User,
        };
        messages.push(Message { role, content: text });
    }

    let (max_tokens, temperature, stop) = match req.generation_config {
        Some(cfg) => (cfg.max_output_tokens, cfg.temperature, cfg.stop_sequences),
        None => (2048, None, vec![]),
    };

    Ok(IntermediateRequest {
        messages,
        max_tokens,
        temperature,
        stop,
        api_key: req.api_key,
    })
}

pub fn from_intermediate(ir: IntermediateResponse) -> serde_json::Value {
    serde_json::json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": ir.content}]
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

pub fn build_worker_body(ir: &IntermediateRequest, _model: &str) -> serde_json::Value {
    let mut parts = Vec::new();
    let mut sys_parts = Vec::new();

    for m in &ir.messages {
        let text_parts: Vec<serde_json::Value> = vec![
            serde_json::json!({"text": m.content.clone()})
        ];
        match m.role {
            Role::System => {
                sys_parts.push(serde_json::json!({"text": m.content}));
            }
            Role::User => {
                parts.push(serde_json::json!({
                    "role": "user",
                    "parts": text_parts,
                }));
            }
            Role::Assistant => {
                parts.push(serde_json::json!({
                    "role": "model",
                    "parts": text_parts,
                }));
            }
        }
    }

    let mut body = serde_json::json!({
        "contents": parts,
        "generationConfig": {
            "maxOutputTokens": ir.max_tokens,
            "temperature": ir.temperature,
            "stopSequences": ir.stop,
        }
    });

    if !sys_parts.is_empty() {
        body["systemInstruction"] = serde_json::json!({
            "parts": sys_parts,
        });
    }

    body
}

pub fn parse_worker_response(
    body: &serde_json::Value,
    model: &str,
    duration_ms: u64,
    cost_usd: Option<f64>,
) -> Result<WorkerResult, OpenFusionError> {
    let candidate = body["candidates"].as_array()
        .and_then(|c| c.first())
        .ok_or_else(|| OpenFusionError::Protocol("missing candidates[0]".into()))?;

    let content = candidate["content"]["parts"].as_array()
        .map(|parts| {
            parts.iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default();

    let finish_reason = candidate["finishReason"]
        .as_str()
        .unwrap_or("STOP")
        .to_string();

    Ok(WorkerResult {
        model: model.to_string(),
        api: "google-generative-ai".into(),
        success: true,
        response: Some(IntermediateResponse {
            content,
            finish_reason,
            usage: Usage {
                prompt_tokens: body["usageMetadata"]["promptTokenCount"].as_u64().unwrap_or(0) as u32,
                completion_tokens: body["usageMetadata"]["candidatesTokenCount"].as_u64().unwrap_or(0) as u32,
                total_tokens: body["usageMetadata"]["totalTokenCount"].as_u64().unwrap_or(0) as u32,
            },
            cost_usd: cost_usd.unwrap_or(0.0),
            duration_ms,
            model: model.to_string(),
        }),
        error: None,
    })
}
