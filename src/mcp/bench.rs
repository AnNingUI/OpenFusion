//! fusion_bench tool: empirical fusion profile benchmarking.

use crate::engine::FusionEngine;
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, Message, Role};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
pub struct BenchParams {
    pub prompt: String,
    pub system: Option<String>,
    /// Number of runs per config (default 1)
    pub runs: Option<u32>,
    /// Fusion profiles to test
    pub configs: Vec<BenchConfig>,
}

#[derive(Debug, Deserialize)]
pub struct BenchConfig {
    pub label: String,
    /// Worker names or model ids. Empty or omitted means all configured workers.
    #[serde(default)]
    pub workers: Vec<String>,
    /// Whether to include judge synthesis in this profile.
    #[serde(default)]
    pub judge_mode: bool,
}

#[derive(Debug, Serialize)]
pub struct BenchResult {
    pub results: Vec<BenchRun>,
    pub ranking: Vec<BenchRanking>,
    pub total_cost_usd: f64,
    pub note: String,
}

#[derive(Debug, Serialize)]
pub struct BenchRun {
    pub label: String,
    pub judge_mode: bool,
    pub runs: u32,
    pub avg_duration_ms: u64,
    pub avg_cost_usd: f64,
    pub worker_count: usize,
    pub avg_success_count: f64,
    pub success_rate: f64,
    pub sample_synthesis: Option<String>,
    pub sample_worker_outputs: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct BenchRanking {
    pub rank: usize,
    pub label: String,
    pub score: f64,
    pub reason: String,
}

pub async fn run(
    engine: Arc<FusionEngine>,
    params: BenchParams,
) -> Result<BenchResult, OpenFusionError> {
    let runs = params.runs.unwrap_or(1).max(1);
    let mut results = Vec::new();

    for bench_cfg in &params.configs {
        let mut total_duration = 0u64;
        let mut total_cost = 0.0;
        let mut total_success_count = 0usize;
        let mut worker_count = 0usize;
        let mut sample_synthesis = None;
        let mut sample_worker_outputs = Vec::new();

        for run_index in 0..runs {
            let request = build_request(&params.prompt, params.system.as_deref());
            let result = engine
                .execute_advisory(&request, &bench_cfg.workers, bench_cfg.judge_mode, false)
                .await?;

            total_duration += result.duration_ms;
            total_cost += result.total_cost_usd;
            total_success_count += result.models_succeeded;
            worker_count = result.worker_results.len();

            if run_index == 0 {
                sample_synthesis = result.synthesis.clone();
                sample_worker_outputs = result
                    .worker_results
                    .iter()
                    .filter_map(|wr| wr.response.as_ref())
                    .map(|resp| resp.content.chars().take(500).collect::<String>())
                    .collect();
            }
        }

        let attempts = (worker_count as u32).saturating_mul(runs).max(1) as f64;
        let success_rate = total_success_count as f64 / attempts;
        results.push(BenchRun {
            label: bench_cfg.label.clone(),
            judge_mode: bench_cfg.judge_mode,
            runs,
            avg_duration_ms: total_duration / runs as u64,
            avg_cost_usd: total_cost / runs as f64,
            worker_count,
            avg_success_count: total_success_count as f64 / runs as f64,
            success_rate,
            sample_synthesis,
            sample_worker_outputs,
        });
    }

    let mut ranking = results
        .iter()
        .map(|result| {
            let latency_penalty = (result.avg_duration_ms as f64 / 1000.0).min(60.0);
            let cost_penalty = (result.avg_cost_usd * 100.0).min(40.0);
            let score = result.success_rate * 100.0 - latency_penalty - cost_penalty;
            BenchRanking {
                rank: 0,
                label: result.label.clone(),
                score,
                reason: format!(
                    "{:.0}% worker success, {}ms avg, ${:.6} avg{}",
                    result.success_rate * 100.0,
                    result.avg_duration_ms,
                    result.avg_cost_usd,
                    if result.judge_mode {
                        ", judge enabled"
                    } else {
                        ""
                    }
                ),
            }
        })
        .collect::<Vec<_>>();
    ranking.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, item) in ranking.iter_mut().enumerate() {
        item.rank = idx + 1;
    }

    Ok(BenchResult {
        total_cost_usd: results
            .iter()
            .map(|result| result.avg_cost_usd * result.runs as f64)
            .sum(),
        results,
        ranking,
        note: "Ranking uses measurable signals only: worker success rate, latency, and reported upstream cost. It does not claim to measure answer quality."
            .into(),
    })
}

fn build_request(prompt: &str, system: Option<&str>) -> IntermediateRequest {
    let content = match system.map(str::trim).filter(|s| !s.is_empty()) {
        Some(system) => format!("# Domain Framing\n{system}\n\n# Task\n{prompt}"),
        None => prompt.to_string(),
    };
    IntermediateRequest {
        messages: vec![Message {
            role: Role::User,
            content,
            tool_call_id: None,
            tool_is_error: None,
            tool_calls: vec![],
        }],
        max_tokens: 2048,
        temperature: Some(0.7),
        top_p: None,
        stop: vec![],
        api_key: None,
        tools: Vec::new(),
        tool_choice: None,
        parallel_tool_calls: None,
        thinking: None,
        session_id: None,
        stream: false,
        system: None,
        metadata: Default::default(),
        native: Default::default(),
    }
}
