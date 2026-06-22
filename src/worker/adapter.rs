//! Protocol-aware HTTP adapter for worker requests.
//!
//! Converts IntermediateRequest → protocol-specific HTTP body,
//! sends to the worker's endpoint, parses protocol-specific response back.

use crate::config::ApiProtocol;
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, WorkerResult};

pub struct WorkerAdapter {
    api: ApiProtocol,
    base_url: String,
    model: String,
    api_key: String,
    web_search: bool,
    web_fetch: bool,
    client: reqwest::Client,
}

impl WorkerAdapter {
    pub fn new(
        api: ApiProtocol,
        base_url: String,
        model: String,
        api_key: String,
        web_search: bool,
        web_fetch: bool,
    ) -> Self {
        Self {
            api,
            base_url,
            model,
            api_key,
            web_search,
            web_fetch,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to build reqwest client"),
        }
    }

    pub async fn execute(
        &self,
        ir: &IntermediateRequest,
        timeout_secs: u64,
    ) -> Result<WorkerResult, OpenFusionError> {
        let mut body = self.build_request_body(ir);

        // Inject OpenRouter plugins for web_search / web_fetch capabilities
        self.inject_plugins(&mut body);

        let path = crate::config::Config::protocol_path(&self.api, &self.model);
        let url = format!("{}{path}", self.base_url.trim_end_matches('/'));

        let start = std::time::Instant::now();

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| OpenFusionError::WorkerTimeout {
            model: self.model.clone(),
            duration_ms: timeout_secs * 1000,
        })?;

        let duration_ms = start.elapsed().as_millis() as u64;

        let response = response.map_err(|e| OpenFusionError::WorkerError {
            model: self.model.clone(),
            message: e.to_string(),
        })?;

        let status = response.status();
        // Extract cost header before consuming the response body
        let cost_usd: Option<f64> = response
            .headers()
            .get("x-openrouter-cost")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok());

        let response_body: serde_json::Value = response.json().await.map_err(|e| {
            OpenFusionError::WorkerError {
                model: self.model.clone(),
                message: format!("Failed to parse response: {e}"),
            }
        })?;

        if !status.is_success() {
            let error_msg = response_body["error"]["message"]
                .as_str()
                .unwrap_or("Unknown API error");
            return Err(OpenFusionError::ApiError {
                status: status.as_u16(),
                body: error_msg.to_string(),
            });
        }

        self.parse_response(&response_body, duration_ms, cost_usd)
    }

    /// Inject OpenRouter plugin definitions for web_search / web_fetch capabilities.
    fn inject_plugins(&self, body: &mut serde_json::Value) {
        let mut plugins: Vec<serde_json::Value> = Vec::new();
        if self.web_search {
            plugins.push(serde_json::json!({"id": "web_search"}));
        }
        if self.web_fetch {
            plugins.push(serde_json::json!({"id": "web_fetch"}));
        }
        if !plugins.is_empty() {
            body["plugins"] = serde_json::Value::Array(plugins);
        }
    }

    fn build_request_body(&self, ir: &IntermediateRequest) -> serde_json::Value {
        match self.api {
            ApiProtocol::OpenaiCompletions => {
                crate::protocol::openai_completions::build_worker_body(ir, &self.model)
            }
            ApiProtocol::OpenaiResponses => {
                crate::protocol::openai_responses::build_worker_body(ir, &self.model)
            }
            ApiProtocol::GoogleGenerativeAi => {
                crate::protocol::google_genai::build_worker_body(ir, &self.model)
            }
            ApiProtocol::AnthropicMessages => {
                crate::protocol::anthropic_messages::build_worker_body(ir, &self.model)
            }
        }
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
                    body, &self.model, duration_ms, cost_usd,
                )
            }
            ApiProtocol::OpenaiResponses => {
                crate::protocol::openai_responses::parse_worker_response(
                    body, &self.model, duration_ms, cost_usd,
                )
            }
            ApiProtocol::GoogleGenerativeAi => {
                crate::protocol::google_genai::parse_worker_response(
                    body, &self.model, duration_ms, cost_usd,
                )
            }
            ApiProtocol::AnthropicMessages => {
                crate::protocol::anthropic_messages::parse_worker_response(
                    body, &self.model, duration_ms, cost_usd,
                )
            }
        }
    }
}
