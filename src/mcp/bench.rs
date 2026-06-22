//! fusion_bench tool: configuration benchmarking.

use crate::config::Config;
use crate::engine::FusionEngine;
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, Message, Role};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
pub struct BenchParams {
    pub prompt: String,
    /// Number of runs per config (default 1)
    pub runs: Option<u32>,
    /// Config profiles to test
    pub configs: Vec<BenchConfig>,
}

#[derive(Debug, Deserialize)]
pub struct BenchConfig {
    pub label: String,
    /// Comma-separated model names to use as workers
    pub workers: String,
    /// Judge model name (optional, from config)
    pub judge: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BenchResult {
    pub results: Vec<BenchRun>,
    pub ranking: Vec<BenchRanking>,
    pub total_cost_usd: f64,
}

#[derive(Debug, Serialize)]
pub struct BenchRun {
    pub label: String,
    pub duration_ms: u64,
    pub cost_usd: f64,
    pub worker_count: usize,
    pub success_count: usize,
    pub synthesis_summary: Option<String>,
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
    _config: Arc<Config>,
    params: BenchParams,
) -> Result<BenchResult, OpenFusionError> {
    let runs = params.runs.unwrap_or(1);
    let mut results = Vec::new();

    for bench_cfg in &params.configs {
        let mut total_duration = 0u64;
        let total_cost = 0.0;
        let mut summaries = Vec::new();
        let mut worker_count = 0usize;
        let mut success_count = 0usize;

        // Build worker model filter from bench config
        let mut model_filter: Vec<String> = if bench_cfg.workers == "all" {
            vec![] // empty means all
        } else {
            bench_cfg.workers.split(',').map(|s| s.trim().to_string()).collect()
        };
        // Include judge model if specified
        if let Some(ref judge) = bench_cfg.judge
            && !model_filter.contains(judge)
            && judge != "all"
        {
            model_filter.push(judge.clone());
        }

        for _ in 0..runs {
            let messages = vec![
                Message { role: Role::User, content: params.prompt.clone() },
            ];
            let request = IntermediateRequest {
                messages,
                max_tokens: 2048,
                temperature: Some(0.7),
                stop: vec![],
                api_key: None,
            };

            // Run workers-only for this config
            let start = std::time::Instant::now();
            let worker_results = engine.execute_workers_filtered(&request, &model_filter).await;
            let duration_ms = start.elapsed().as_millis() as u64;

            total_duration += duration_ms;
            worker_count = worker_results.len();
            success_count = worker_results.iter().filter(|w| w.success).count();

            // Collect summaries
            for wr in &worker_results {
                if let Some(ref resp) = wr.response {
                    summaries.push(resp.content.chars().take(500).collect::<String>());
                }
            }
        }

        results.push(BenchRun {
            label: bench_cfg.label.clone(),
            duration_ms: total_duration / runs as u64,
            cost_usd: total_cost / runs as f64,
            worker_count,
            success_count,
            synthesis_summary: summaries.first().cloned(),
        });
    }

    // Build ranking by success ratio and speed
    let mut ranking: Vec<BenchRanking> = results.iter().map(|r| {
        let score = (r.success_count as f64 / r.worker_count.max(1) as f64) * 100.0
            - (r.duration_ms as f64 / 1000.0).min(50.0);
        BenchRanking {
            rank: 0,
            label: r.label.clone(),
            score,
            reason: format!(
                "{} workers, {} succeeded, {}ms avg",
                r.worker_count, r.success_count, r.duration_ms
            ),
        }
    }).collect();

    ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    for (i, r) in ranking.iter_mut().enumerate() {
        r.rank = i + 1;
    }

    Ok(BenchResult {
        total_cost_usd: results.iter().map(|r| r.cost_usd).sum(),
        results,
        ranking,
    })
}
