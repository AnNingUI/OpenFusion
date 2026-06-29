//! Fusion engine for MCP advisory fanout and optional judge synthesis.

use crate::config::Config;
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, Message, Role, WorkerResult};
use crate::worker::Worker;
use std::sync::Arc;
use std::time::Duration;

pub mod session;
pub use session::SessionStore;

pub struct FusionEngine {
    pub config: Arc<Config>,
    workers: Vec<Worker>,
    judge_worker: Worker,
    session_store: Arc<SessionStore>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FusionResult {
    pub session_id: String,
    pub judge_response: Option<WorkerResult>,
    pub worker_results: Vec<WorkerResult>,
    pub enhanced_prompt_used: bool,
    pub synthesis: Option<String>,
    pub consensus: Vec<String>,
    pub contradictions: Vec<Contradiction>,
    pub blind_spots: Vec<String>,
    pub judge_output_raw: Option<String>,
    pub passthrough_tool_calls: Vec<crate::protocol::ToolCall>,
    pub total_cost_usd: f64,
    pub models_succeeded: usize,
    pub models_failed: usize,
    pub duration_ms: u64,
    pub timestamp: String,
    pub original_prompt: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Contradiction {
    pub topic: String,
    pub views: Vec<ContradictionView>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContradictionView {
    pub model: String,
    pub position: String,
}

impl FusionEngine {
    pub fn new(config: Arc<Config>) -> Result<Self, OpenFusionError> {
        let judge_api_key = config
            .resolve_api_key(&config.judge)
            .ok_or(OpenFusionError::NoApiKey)?;

        let judge_worker = Worker::new(
            config.judge.model.clone(),
            config.judge.api,
            config.judge.base_url.clone(),
            judge_api_key,
        );

        let mut workers = Vec::new();
        for wcfg in &config.workers {
            let api_key = config.resolve_api_key(wcfg).ok_or_else(|| {
                OpenFusionError::Config(format!("No API key for worker '{}'", wcfg.model))
            })?;
            let name = wcfg.name.clone().unwrap_or_else(|| wcfg.model.clone());
            workers.push(
                Worker::new(wcfg.model.clone(), wcfg.api, wcfg.base_url.clone(), api_key)
                    .with_name(name)
                    .with_advisory_prompt(wcfg.personality.clone()),
            );
        }

        let session_store = Arc::new(SessionStore::new(
            config.storage.sessions_dir.clone(),
            config.storage.max_sessions,
        )?);

        Ok(Self {
            config,
            workers,
            judge_worker,
            session_store,
        })
    }

    pub async fn execute_advisory(
        &self,
        request: &IntermediateRequest,
        model_filter: &[String],
        judge_mode: bool,
        save_session: bool,
    ) -> Result<FusionResult, OpenFusionError> {
        let start = std::time::Instant::now();
        let session_id = uuid::Uuid::new_v4().to_string();
        let original_prompt = request
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::User))
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let worker_results = self.execute_workers_filtered(request, model_filter).await;
        let succeeded_count = worker_results.iter().filter(|w| w.success).count();
        let failed = worker_results.len().saturating_sub(succeeded_count);

        if succeeded_count < self.config.fusion.min_workers {
            let failures = worker_results
                .iter()
                .filter(|w| !w.success)
                .map(|w| format!("  {}: {}", w.model, w.error.as_deref().unwrap_or("unknown")))
                .collect::<Vec<_>>()
                .join("\n");
            return Err(OpenFusionError::InsufficientWorkers {
                success_count: succeeded_count,
                total: worker_results.len(),
                min: self.config.fusion.min_workers,
                details: failures,
            });
        }

        let worker_cost = worker_results
            .iter()
            .filter_map(|w| w.response.as_ref().map(|r| r.cost_usd))
            .sum::<f64>();
        let mut judge_response = None;
        let mut judge_output_raw = None;
        let mut passthrough_tool_calls = Vec::new();
        let mut total_cost_usd = worker_cost;

        if judge_mode {
            let mut judge_request = request.clone();
            if let Some(ctx) = self.build_enhanced_context(&worker_results) {
                judge_request.messages.insert(
                    0,
                    Message {
                        role: Role::System,
                        content: format!(
                            "{ctx}\n\n# Judge Task\nSynthesize the worker proposals into the most concise, highest-signal answer. Keep strong points, discard weak points, call out risks, and avoid pretending to know repository details not present in the task."
                        ),
                        tool_call_id: None,
                        tool_is_error: None,
                        tool_calls: vec![],
                    },
                );
            }
            let jr = self.route_to_judge(&judge_request).await?;
            if let Some(resp) = jr.response.as_ref() {
                total_cost_usd += resp.cost_usd;
                judge_output_raw = Some(resp.content.clone());
                passthrough_tool_calls = resp.tool_calls.clone();
            }
            judge_response = Some(jr);
        }

        let result = FusionResult {
            session_id,
            judge_response,
            worker_results,
            enhanced_prompt_used: judge_mode,
            synthesis: judge_output_raw.clone(),
            consensus: vec![],
            contradictions: vec![],
            blind_spots: vec![],
            judge_output_raw,
            passthrough_tool_calls,
            total_cost_usd,
            models_succeeded: succeeded_count,
            models_failed: failed,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
            original_prompt,
        };

        if save_session {
            let _ = self.session_store.save(&result).await;
        }

        Ok(result)
    }

    pub async fn execute_workers_only(&self, request: &IntermediateRequest) -> Vec<WorkerResult> {
        let mut advisory_request = request.clone();
        advisory_request.tools.clear();
        advisory_request.tool_choice = None;
        advisory_request.parallel_tool_calls = None;
        self.dispatch_workers(&advisory_request).await
    }

    pub async fn execute_workers_filtered(
        &self,
        request: &IntermediateRequest,
        model_filter: &[String],
    ) -> Vec<WorkerResult> {
        let use_all = model_filter.is_empty() || model_filter.iter().any(|m| m == "all");
        if use_all {
            return self.execute_workers_only(request).await;
        }

        let mut advisory_request = request.clone();
        advisory_request.tools.clear();
        advisory_request.tool_choice = None;
        advisory_request.parallel_tool_calls = None;
        let filtered = self
            .workers
            .iter()
            .filter(|w| model_filter.iter().any(|m| m == &w.model || m == &w.name))
            .cloned()
            .collect::<Vec<_>>();
        self.dispatch_specific_workers(&filtered, &advisory_request)
            .await
    }

    pub fn sessions(&self) -> &Arc<SessionStore> {
        &self.session_store
    }

    async fn route_to_judge(
        &self,
        request: &IntermediateRequest,
    ) -> Result<WorkerResult, OpenFusionError> {
        let max_retries = self
            .config
            .fusion
            .retry
            .max(crate::error::DEFAULT_TRANSIENT_RETRIES);
        let mut last_err = None;

        for attempt in 1..=max_retries + 1 {
            match self
                .judge_worker
                .execute(request, self.config.fusion.timeout_secs)
                .await
            {
                Ok(result) => return Ok(result),
                Err(e) if attempt <= max_retries && is_retryable(&e) => {
                    let delay = retry_delay(&e, attempt);
                    tracing::warn!(
                        "Judge attempt {attempt}/{} failed: {e} - retrying in {}ms",
                        max_retries + 1,
                        delay.as_millis()
                    );
                    tokio::time::sleep(delay).await;
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_err
            .unwrap_or_else(|| OpenFusionError::Config("judge retry loop did not run".into())))
    }

    fn build_enhanced_context(&self, worker_results: &[WorkerResult]) -> Option<String> {
        let succeeded = worker_results
            .iter()
            .filter(|w| w.success)
            .collect::<Vec<_>>();
        if succeeded.is_empty() {
            return None;
        }

        let mut ctx = String::with_capacity(4096);
        ctx.push_str("# Worker Proposals\n\n");
        ctx.push_str(
            "Multiple AI workers analyzed the task in parallel. Below are their proposals.\n",
        );
        ctx.push_str("Use these as reference: absorb strengths, discard weaknesses.\n\n");

        for (i, wr) in succeeded.iter().enumerate() {
            if let Some(resp) = wr.response.as_ref()
                && !resp.content.is_empty()
            {
                ctx.push_str(&format!(
                    "## Worker {} ({})\n{}\n\n",
                    i + 1,
                    wr.name,
                    resp.content
                ));
            }
        }

        Some(ctx)
    }

    async fn dispatch_workers(&self, request: &IntermediateRequest) -> Vec<WorkerResult> {
        self.dispatch_specific_workers(&self.workers, request).await
    }

    async fn dispatch_specific_workers(
        &self,
        workers: &[Worker],
        request: &IntermediateRequest,
    ) -> Vec<WorkerResult> {
        let mut set = tokio::task::JoinSet::new();
        let timeout = self.config.fusion.worker_timeout_secs;
        let retries = self.config.fusion.retry;

        for (i, worker) in workers.iter().enumerate() {
            if i > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            }
            let worker = worker.clone();
            let request = request.clone();
            let model = worker.model.clone();
            let name = worker.name.clone();
            set.spawn(async move {
                match Self::execute_with_retry(worker, request, timeout, retries).await {
                    Ok(result) => result,
                    Err(e) => WorkerResult {
                        model,
                        name,
                        api: "error".into(),
                        success: false,
                        response: None,
                        error: Some(e.to_string()),
                    },
                }
            });
        }

        Self::collect_join_set(&mut set).await
    }

    async fn execute_with_retry(
        worker: Worker,
        request: IntermediateRequest,
        timeout: u64,
        retries: u32,
    ) -> Result<WorkerResult, OpenFusionError> {
        let max_retries = retries.max(crate::error::DEFAULT_TRANSIENT_RETRIES);
        let mut last_err = None;

        for attempt in 1..=max_retries + 1 {
            match worker.execute(&request, timeout).await {
                Ok(result) => return Ok(result),
                Err(e) if attempt <= max_retries && is_retryable(&e) => {
                    let delay = retry_delay(&e, attempt);
                    tracing::warn!(
                        "Worker '{}' attempt {attempt}/{} failed: {e} - retrying in {}ms",
                        worker.model,
                        max_retries + 1,
                        delay.as_millis()
                    );
                    tokio::time::sleep(delay).await;
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_err
            .unwrap_or_else(|| OpenFusionError::Config("worker retry loop did not run".into())))
    }

    async fn collect_join_set(set: &mut tokio::task::JoinSet<WorkerResult>) -> Vec<WorkerResult> {
        let mut results = Vec::new();
        while let Some(outcome) = set.join_next().await {
            match outcome {
                Ok(result) => results.push(result),
                Err(join_err) => results.push(WorkerResult {
                    model: "worker-panic".into(),
                    name: "worker-panic".into(),
                    api: "unknown".into(),
                    success: false,
                    response: None,
                    error: Some(format!("Worker panicked: {join_err}")),
                }),
            }
        }
        results
    }
}

fn is_retryable(err: &OpenFusionError) -> bool {
    match err {
        OpenFusionError::ApiError { status, body, .. } => {
            crate::error::is_transient_upstream_error(*status, body)
        }
        OpenFusionError::WorkerTimeout { .. } => true,
        OpenFusionError::Network(_) => true,
        _ => false,
    }
}

fn retry_delay(err: &OpenFusionError, attempt: u32) -> Duration {
    if let Some(secs) = err.retry_after_secs() {
        return Duration::from_secs(secs.clamp(1, 10));
    }
    let millis = 500_u64.saturating_mul(1_u64 << attempt.saturating_sub(1).min(4));
    Duration::from_millis(millis.min(8_000))
}
