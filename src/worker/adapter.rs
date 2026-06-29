//! Protocol-aware HTTP adapter for single-shot worker requests.

use crate::config::ApiProtocol;
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, Message, Role, WorkerResult};
use std::sync::Arc;

fn extract_upstream_error_message(status: reqwest::StatusCode, response_text: &str) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(response_text) {
        if let Some(message) = value
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(|message| message.as_str())
        {
            return message.to_string();
        }
        if let Some(error) = value.get("error").and_then(|error| error.as_str()) {
            return error.to_string();
        }
        if let Some(message) = value.get("message").and_then(|message| message.as_str()) {
            return message.to_string();
        }
        if let Some(detail) = value.get("detail").and_then(|detail| detail.as_str()) {
            return detail.to_string();
        }
    }

    let raw = response_text.trim();
    if raw.is_empty() {
        format!("HTTP {status}")
    } else {
        raw.chars().take(1000).collect()
    }
}

#[derive(Clone)]
pub struct Worker {
    pub model: String,
    pub name: String,
    pub api: ApiProtocol,
    pub base_url: String,
    api_key: String,
    client: reqwest::Client,
    advisory_prompt: Option<String>,
    gemini_shadow: Arc<crate::protocol::gemini_shadow::GeminiShadowStore>,
}

impl Worker {
    pub fn new(model: String, api: ApiProtocol, base_url: String, api_key: String) -> Self {
        Self {
            name: model.clone(),
            model,
            api,
            base_url,
            api_key,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to build reqwest client"),
            advisory_prompt: None,
            gemini_shadow: Arc::new(crate::protocol::gemini_shadow::GeminiShadowStore::default()),
        }
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn with_advisory_prompt(mut self, personality: Vec<String>) -> Self {
        self.advisory_prompt = Some(build_advisory_prompt(&personality));
        self
    }

    /// Execute a single model request.
    ///
    /// OpenFusion no longer executes model tool calls locally. Any returned tool
    /// calls are preserved in the response for the MCP host/client to handle.
    pub async fn execute(
        &self,
        ir: &IntermediateRequest,
        timeout_secs: u64,
    ) -> Result<WorkerResult, OpenFusionError> {
        let request = self.prepare_request(ir);
        self.log_request("[TO-MODEL]", &request);

        let start = std::time::Instant::now();
        let provider_id = format!("{}:{}", self.api_str(), self.model);
        let session_id = request.session_id.clone();
        let gemini_shadow_turns = session_id
            .as_deref()
            .and_then(|session_id| {
                self.gemini_shadow
                    .get_session(&provider_id, session_id)
                    .map(|snapshot| snapshot.turns)
            })
            .unwrap_or_default();
        let body = self.build_request_body_with_gemini_shadow(&request, &gemini_shadow_turns);
        let (response_text, cost_usd) = self.post_json(&body, timeout_secs).await?;
        let mut response_body: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| OpenFusionError::WorkerError {
                model: self.model.clone(),
                message: format!(
                    "Failed to parse JSON response: {e}\n--- raw body ---\n{}",
                    response_text.chars().take(500).collect::<String>(),
                ),
            })?;

        if self.api == ApiProtocol::GoogleGenerativeAi {
            crate::protocol::google_genai::rectify_response_missing_function_call_ids(
                &mut response_body,
            );
        }

        let mut result =
            self.parse_response(&response_body, start.elapsed().as_millis() as u64, cost_usd)?;
        result.name = self.name.clone();

        if self.api == ApiProtocol::GoogleGenerativeAi
            && let Some(turn) =
                crate::protocol::google_genai::extract_assistant_shadow_turn(&response_body)
            && !turn.tool_calls.is_empty()
            && let Some(session_id) = session_id.as_deref()
        {
            self.gemini_shadow.record_assistant_turn(
                provider_id,
                session_id,
                turn.assistant_content,
                turn.tool_calls,
            );
        }

        tracing::info!(
            model = %result.model,
            name = %result.name,
            success = result.success,
            "worker completed"
        );
        Ok(result)
    }

    fn prepare_request(&self, ir: &IntermediateRequest) -> IntermediateRequest {
        let mut request = ir.clone();
        if let Some(prompt) = &self.advisory_prompt {
            request.messages.retain(|m| !matches!(m.role, Role::System));
            request.messages.insert(
                0,
                Message {
                    role: Role::System,
                    content: prompt.clone(),
                    tool_call_id: None,
                    tool_is_error: None,
                    tool_calls: vec![],
                },
            );
            request.tools.clear();
            request.tool_choice = None;
            request.parallel_tool_calls = None;
        }
        request.stream = false;
        request
    }

    async fn post_json(
        &self,
        body: &serde_json::Value,
        timeout_secs: u64,
    ) -> Result<(String, Option<f64>), OpenFusionError> {
        let path = crate::config::Config::protocol_path(&self.api, &self.model);
        let base = self.base_url.trim_end_matches('/');
        let base = if path.starts_with("/v1") && base.ends_with("/v1") {
            &base[..base.len() - 3]
        } else {
            base
        };
        let url = format!("{base}{path}");

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(body)
                .send(),
        )
        .await
        .map_err(|_| OpenFusionError::WorkerTimeout {
            model: self.model.clone(),
            duration_ms: timeout_secs * 1000,
        })??;

        let status = response.status();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        let cost_usd = response
            .headers()
            .get("x-openrouter-cost")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok());
        let response_text = response
            .text()
            .await
            .map_err(|e| OpenFusionError::WorkerError {
                model: self.model.clone(),
                message: format!("Failed to read response body: {e}"),
            })?;

        if !status.is_success() {
            let error_msg = extract_upstream_error_message(status, &response_text);
            if status.as_u16() == 400 {
                tracing::warn!(
                    model = %self.model,
                    api = %self.api_str(),
                    status = status.as_u16(),
                    error = %error_msg,
                    request_preview = %body.to_string().chars().take(3000).collect::<String>(),
                    response_preview = %response_text.chars().take(1000).collect::<String>(),
                    "Upstream API rejected request"
                );
            }
            return Err(OpenFusionError::ApiError {
                status: status.as_u16(),
                body: error_msg,
                retry_after_secs: retry_after,
            });
        }

        Ok((response_text, cost_usd))
    }

    fn api_str(&self) -> &'static str {
        match self.api {
            ApiProtocol::OpenaiCompletions => "openai-completions",
            ApiProtocol::OpenaiResponses => "openai-responses",
            ApiProtocol::GoogleGenerativeAi => "google-generative-ai",
            ApiProtocol::AnthropicMessages => "anthropic-messages",
        }
    }

    fn build_request_body_with_gemini_shadow(
        &self,
        ir: &IntermediateRequest,
        gemini_shadow_turns: &[crate::protocol::gemini_shadow::GeminiAssistantTurn],
    ) -> serde_json::Value {
        let body = match self.api {
            ApiProtocol::OpenaiCompletions => {
                crate::protocol::openai_completions::build_worker_body(ir, &self.model)
            }
            ApiProtocol::OpenaiResponses => {
                crate::protocol::openai_responses::build_worker_body(ir, &self.model)
            }
            ApiProtocol::GoogleGenerativeAi => {
                crate::protocol::google_genai::build_worker_body_with_shadow(
                    ir,
                    &self.model,
                    gemini_shadow_turns,
                )
            }
            ApiProtocol::AnthropicMessages => {
                crate::protocol::anthropic_messages::build_worker_body(ir, &self.model)
            }
        };
        self.log_body(ir, &body);
        body
    }

    fn parse_response(
        &self,
        body: &serde_json::Value,
        duration_ms: u64,
        cost_usd: Option<f64>,
    ) -> Result<WorkerResult, OpenFusionError> {
        match self.api {
            ApiProtocol::OpenaiCompletions => {
                crate::protocol::openai_completions::parse_worker_response(
                    body,
                    &self.model,
                    duration_ms,
                    cost_usd,
                )
            }
            ApiProtocol::OpenaiResponses => {
                crate::protocol::openai_responses::parse_worker_response(
                    body,
                    &self.model,
                    duration_ms,
                    cost_usd,
                )
            }
            ApiProtocol::GoogleGenerativeAi => {
                crate::protocol::google_genai::parse_worker_response(
                    body,
                    &self.model,
                    duration_ms,
                    cost_usd,
                )
            }
            ApiProtocol::AnthropicMessages => {
                crate::protocol::anthropic_messages::parse_worker_response(
                    body,
                    &self.model,
                    duration_ms,
                    cost_usd,
                )
            }
        }
    }

    fn log_request(&self, tag: &str, ir: &IntermediateRequest) {
        tracing::info!(
            "{} api={} model={} msgs={} tools={}",
            tag,
            self.api_str(),
            self.model,
            ir.messages.len(),
            ir.tools.len(),
        );
        for (i, message) in ir.messages.iter().enumerate() {
            let preview: String = message.content.chars().take(200).collect();
            tracing::info!(
                "{} msg[{i}] role={:?} content_len={} tc={} preview={:?}",
                tag,
                message.role,
                message.content.len(),
                message.tool_calls.len(),
                preview
            );
        }
    }

    fn log_body(&self, ir: &IntermediateRequest, body: &serde_json::Value) {
        let input_preview = match self.api {
            ApiProtocol::OpenaiResponses => body
                .get("input")
                .map(|v| v.to_string().chars().take(500).collect::<String>())
                .unwrap_or_default(),
            ApiProtocol::OpenaiCompletions | ApiProtocol::AnthropicMessages => body
                .get("messages")
                .map(|v| v.to_string().chars().take(500).collect::<String>())
                .unwrap_or_default(),
            ApiProtocol::GoogleGenerativeAi => body
                .get("contents")
                .map(|v| v.to_string().chars().take(500).collect::<String>())
                .unwrap_or_default(),
        };
        let tag = if self.advisory_prompt.is_some() {
            "[WORKER-BODY]"
        } else {
            "[JUDGE-BODY]"
        };
        tracing::info!(
            "{} api={} model={} stream={} input/messages preview: {}",
            tag,
            self.api_str(),
            self.model,
            ir.stream,
            input_preview
        );
    }
}

fn build_advisory_prompt(personality: &[String]) -> String {
    let mut prompt = String::from(
        "<worker-role>\n\
You are an advisory worker in a multi-model fusion panel.\n\
You do not have access to the user's real filesystem, repository, shell, browser, or tools.\n\
The calling AI client is responsible for reading files, abstracting project details, and implementing changes.\n\n\
Your job:\n\
- Analyze only the self-contained abstract task you receive.\n\
- Provide concrete recommendations, tradeoffs, risks, and implementation guidance.\n\
- Do not ask to inspect files or call tools.\n\
- Do not invent repository-specific facts that were not provided.\n\
- Prefer actionable engineering reasoning over generic advice.\n",
    );
    if !personality.is_empty() {
        prompt.push_str("\nWorking style: ");
        prompt.push_str(&personality.join(", "));
        prompt.push_str(".\n");
    }
    prompt.push_str("</worker-role>");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_error_message_extracts_detail_field() {
        let status = reqwest::StatusCode::BAD_REQUEST;
        let body = r#"{"detail":"The 'gpt-5.4' model is not supported when using Codex with a ChatGPT account."}"#;

        assert_eq!(
            extract_upstream_error_message(status, body),
            "The 'gpt-5.4' model is not supported when using Codex with a ChatGPT account."
        );
    }
}
