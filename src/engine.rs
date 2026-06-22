//! Fusion engine: parallel worker dispatch + judge synthesis.
//!
//! Core flow:
//!   1. Spawn all workers in parallel (JoinSet)
//!   2. Collect results, check min_workers threshold
//!   3. Build judge prompt from successful responses
//!   4. Judge synthesizes final answer
//!   5. If judge fails, fall back to raw worker responses

use crate::config::Config;
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, WorkerResult};
use crate::worker::Worker;
use std::sync::Arc;
use std::time::Duration;

pub mod judge;
pub use judge::{build_judge_prompt, parse_judge_output};

pub mod session;
pub use session::SessionStore;

pub struct FusionEngine {
    pub config: Arc<Config>,
    workers: Vec<Worker>,
    judge_worker: Worker,
    session_store: Arc<SessionStore>,
}

/// The result of a single fusion execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FusionResult {
    pub session_id: String,
    pub synthesis: Option<String>,
    pub consensus: Vec<String>,
    pub contradictions: Vec<Contradiction>,
    pub blind_spots: Vec<String>,
    pub worker_results: Vec<WorkerResult>,
    pub judge_output_raw: Option<String>,
    pub total_cost_usd: f64,
    pub models_succeeded: usize,
    pub models_failed: usize,
    pub duration_ms: u64,
    pub timestamp: String,
    /// Original user prompt text (for session replay)
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
            config.judge.api.clone(),
            config.judge.base_url.clone(),
            judge_api_key,
            config.judge.web_search,
            config.judge.web_fetch,
        );

        let mut workers = Vec::new();
        for wcfg in &config.workers {
            let api_key = config
                .resolve_api_key(wcfg)
                .ok_or_else(|| OpenFusionError::Config(format!(
                    "No API key for worker '{}'", wcfg.model
                )))?;
            workers.push(Worker::new(
                wcfg.model.clone(),
                wcfg.api.clone(),
                wcfg.base_url.clone(),
                api_key,
                wcfg.web_search,
                wcfg.web_fetch,
            ));
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

    /// Execute full fusion: parallel workers → judge synthesis.
    pub async fn execute(
        &self,
        request: &IntermediateRequest,
    ) -> Result<FusionResult, OpenFusionError> {
        let start = std::time::Instant::now();
        let session_id = uuid::Uuid::new_v4().to_string();

        // ── Phase 1: Parallel worker dispatch ──
        let worker_results = self.dispatch_workers(request).await;

        // Extract original user prompt for session replay
        let original_prompt = request
            .messages
            .iter()
            .filter(|m| matches!(m.role, crate::protocol::Role::User))
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let succeeded: Vec<_> = worker_results.iter().filter(|w| w.success).collect();
        let failed = worker_results.len() - succeeded.len();
        let succeeded_count = succeeded.len();

        // Check min_workers threshold
        if succeeded.len() < self.config.fusion.min_workers {
            return Err(OpenFusionError::InsufficientWorkers {
                success_count: succeeded.len(),
                total: worker_results.len(),
                min: self.config.fusion.min_workers,
            });
        }

        // ── Phase 2: Judge synthesis ──
        let (synthesis, consensus, contradictions, blind_spots, judge_raw) =
            self.run_judge(request, &succeeded).await;

        let duration_ms = start.elapsed().as_millis() as u64;

        let total_cost_usd: f64 = worker_results.iter()
            .filter_map(|w| w.response.as_ref().map(|r| r.cost_usd))
            .sum();

        let result = FusionResult {
            session_id: session_id.clone(),
            synthesis,
            consensus,
            contradictions,
            blind_spots,
            worker_results,
            judge_output_raw: judge_raw,
            total_cost_usd,
            models_succeeded: succeeded_count,
            models_failed: failed,
            duration_ms,
            timestamp: chrono::Utc::now().to_rfc3339(),
            original_prompt,
        };

        // ── Phase 3: Save session ──
        let _ = self.session_store.save(&result).await;

        Ok(result)
    }

    /// Execute workers only — no judge. Used by fusion_diff and fusion_bench.
    pub async fn execute_workers_only(
        &self,
        request: &IntermediateRequest,
    ) -> Vec<WorkerResult> {
        self.dispatch_workers(request).await
    }

    /// Execute only workers whose model names match the filter.
    /// If filter is empty or contains "all", dispatches all workers.
    /// If filter contains model names, only those workers are used.
    pub async fn execute_workers_filtered(
        &self,
        request: &IntermediateRequest,
        model_filter: &[String],
    ) -> Vec<WorkerResult> {
        let use_all = model_filter.is_empty() || model_filter.iter().any(|m| m == "all");
        if use_all {
            return self.execute_workers_only(request).await;
        }
        let filtered: Vec<Worker> = self.workers
            .iter()
            .filter(|w| model_filter.iter().any(|m| m == &w.model))
            .cloned()
            .collect();
        self.dispatch_specific_workers(&filtered, request).await
    }

    /// Dispatch a specific set of workers (used for filtered dispatch + judge inclusion).
    async fn dispatch_specific_workers(
        &self,
        workers: &[Worker],
        request: &IntermediateRequest,
    ) -> Vec<WorkerResult> {
        let mut set = tokio::task::JoinSet::new();
        let timeout = self.config.fusion.worker_timeout_secs;
        let retries = self.config.fusion.retry;

        for worker in workers {
            let w = worker.clone();
            let ir = request.clone();
            set.spawn(async move {
                Self::execute_with_retry(w, ir, timeout, retries).await
            });
        }

        Self::collect_join_set(&mut set).await
    }

    /// Get session store reference for MCP tools.
    pub fn sessions(&self) -> &Arc<SessionStore> {
        &self.session_store
    }

    /// Get workers reference for bench tool.
    pub fn workers(&self) -> &[Worker] {
        &self.workers
    }

    /// Get judge worker reference.
    pub fn judge(&self) -> &Worker {
        &self.judge_worker
    }

    // ── Internal ──

    async fn dispatch_workers(&self, request: &IntermediateRequest) -> Vec<WorkerResult> {
        let mut set = tokio::task::JoinSet::new();
        let timeout = self.config.fusion.worker_timeout_secs;
        let retries = self.config.fusion.retry;

        for worker in &self.workers {
            let w = worker.clone();
            let ir = request.clone();
            set.spawn(async move {
                Self::execute_with_retry(w, ir, timeout, retries).await
            });
        }

        Self::collect_join_set(&mut set).await
    }

    async fn execute_with_retry(
        worker: Worker,
        ir: IntermediateRequest,
        timeout: u64,
        retries: u32,
    ) -> Result<WorkerResult, OpenFusionError> {
        let mut last_err: Option<OpenFusionError> = None;
        for attempt in 1..=retries + 1 {
            match worker.execute(&ir, timeout).await {
                Ok(r) if r.success => return Ok(r),
                Ok(r) => return Ok(r),
                Err(e) => {
                    if attempt <= retries && is_retryable(&e) {
                        let delay = Duration::from_millis(500 * attempt as u64);
                        tracing::warn!(
                            "Worker '{}' attempt {attempt}/{} failed: {e} — retrying in {}ms",
                            worker.model,
                            retries + 1,
                            delay.as_millis()
                        );
                        tokio::time::sleep(delay).await;
                        last_err = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        Err(last_err.unwrap())
    }

    async fn collect_join_set(
        set: &mut tokio::task::JoinSet<Result<crate::protocol::WorkerResult, OpenFusionError>>,
    ) -> Vec<WorkerResult> {
        let mut results = Vec::new();
        while let Some(outcome) = set.join_next().await {
            match outcome {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => {
                    results.push(WorkerResult {
                        model: "unknown".into(),
                        api: "unknown".into(),
                        success: false,
                        response: None,
                        error: Some(e.to_string()),
                    });
                }
                Err(join_err) => {
                    results.push(WorkerResult {
                        model: "unknown".into(),
                        api: "unknown".into(),
                        success: false,
                        response: None,
                        error: Some(format!("Worker panicked: {join_err}")),
                    });
                }
            }
        }
        results
    }

    async fn run_judge(
        &self,
        request: &IntermediateRequest,
        worker_results: &[&WorkerResult],
    ) -> (
        Option<String>,
        Vec<String>,
        Vec<Contradiction>,
        Vec<String>,
        Option<String>,
    ) {
        let judge_prompt = build_judge_prompt(request, worker_results);

        let judge_ir = IntermediateRequest {
            messages: vec![crate::protocol::Message {
                role: crate::protocol::Role::User,
                content: judge_prompt,
            }],
            max_tokens: 4096,
            temperature: Some(0.3),
            stop: vec![],
            api_key: None,
        };

        match self.judge_worker.execute(&judge_ir, self.config.fusion.timeout_secs).await {
            Ok(result) if result.success => {
                let content = result.response.as_ref()
                    .map(|r| r.content.clone())
                    .unwrap_or_default();
                match parse_judge_output(&content) {
                    Ok(judge_out) => (
                        Some(judge_out.synthesis),
                        judge_out.consensus,
                        judge_out.contradictions,
                        judge_out.blind_spots,
                        Some(content),
                    ),
                    Err(_) => (
                        Some(format!("[Judge parse failed, raw output]\n\n{content}")),
                        vec![],
                        vec![],
                        vec![],
                        Some(content),
                    ),
                }
            }
            Ok(_) | Err(_) => {
                // Degrade: return raw worker concatenation
                let raw = worker_results.iter()
                    .filter_map(|w| w.response.as_ref())
                    .map(|r| format!("## {}\n\n{}", r.model, r.content))
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n");
                (
                    Some(format!("[Judge unavailable — raw worker responses]\n\n{raw}")),
                    vec![],
                    vec![],
                    vec![],
                    None,
                )
            }
        }
    }
}

/// Determine whether an error is retryable (5xx server errors, timeouts).
/// 4xx client errors and parse errors are not retryable.
fn is_retryable(err: &OpenFusionError) -> bool {
    match err {
        OpenFusionError::ApiError { status, .. } => *status >= 500,
        OpenFusionError::WorkerTimeout { .. } => true,
        OpenFusionError::Network(_) => true,
        _ => false,
    }
}
