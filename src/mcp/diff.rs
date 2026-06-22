//! fusion_diff tool: raw model comparison without synthesis.

use crate::engine::{Contradiction, ContradictionView, FusionEngine, FusionResult};
use crate::error::OpenFusionError;
use crate::protocol::{IntermediateRequest, Message, Role};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
pub struct DiffParams {
    pub prompt: String,
    pub system: Option<String>,
    pub panel_override: Option<String>,
    pub save_session: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct DiffResult {
    pub session_id: String,
    pub responses: Vec<WorkerResponseView>,
    pub comparison: DiffComparison,
    pub cost_usd: f64,
    pub duration_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct WorkerResponseView {
    pub model: String,
    pub api: String,
    pub content: String,
    pub token_count: u32,
    pub duration_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DiffComparison {
    pub common_points: Vec<String>,
    pub unique_insights: Vec<ModelInsight>,
    pub disagreements: Vec<Disagreement>,
    pub coverage_gaps: Vec<String>,
    pub best_by_aspect: Vec<AspectRanking>,
}

#[derive(Debug, Serialize)]
pub struct ModelInsight {
    pub model: String,
    pub insight: String,
}

#[derive(Debug, Serialize)]
pub struct Disagreement {
    pub topic: String,
    pub views: Vec<DisagreementView>,
}

#[derive(Debug, Serialize)]
pub struct DisagreementView {
    pub model: String,
    pub position: String,
}

#[derive(Debug, Serialize)]
pub struct AspectRanking {
    pub aspect: String,
    pub best_model: String,
    pub reason: String,
}

pub async fn run(engine: Arc<FusionEngine>, params: DiffParams) -> Result<DiffResult, OpenFusionError> {
    let mut messages = Vec::new();
    if let Some(sys) = &params.system {
        messages.push(Message { role: Role::System, content: sys.clone() });
    }
    messages.push(Message { role: Role::User, content: params.prompt.clone() });

    let request = IntermediateRequest {
        messages,
        max_tokens: 2048,
        temperature: Some(0.7),
        stop: vec![],
        api_key: None,
    };

    let start = std::time::Instant::now();

    // Apply panel_override: filter workers to only those specified
    let model_filter: Vec<String> = params.panel_override.as_ref()
        .map(|s| s.split(',').map(|m| m.trim().to_string()).collect())
        .unwrap_or_default();
    let worker_results = engine.execute_workers_filtered(&request, &model_filter).await;
    let duration_ms = start.elapsed().as_millis() as u64;
    let session_id = uuid::Uuid::new_v4().to_string();

    let responses: Vec<WorkerResponseView> = worker_results.iter().map(|wr| {
        WorkerResponseView {
            model: wr.model.clone(),
            api: wr.api.clone(),
            content: wr.response.as_ref().map(|r| r.content.clone()).unwrap_or_default(),
            token_count: wr.response.as_ref().map(|r| r.usage.total_tokens).unwrap_or(0),
            duration_ms: wr.response.as_ref().map(|r| r.duration_ms).unwrap_or(0),
            error: wr.error.clone(),
        }
    }).collect();

    let comparison = build_comparison(&responses);

    // Compute total cost from worker responses (must be BEFORE save_session moves worker_results)
    let total_cost = worker_results.iter()
        .filter_map(|w| w.response.as_ref().map(|r| r.cost_usd))
        .sum();

    // Save session if requested
    if params.save_session.unwrap_or(false) {
        let result = FusionResult {
            session_id: session_id.clone(),
            synthesis: None,
            consensus: comparison.common_points.clone(),
            contradictions: comparison.disagreements.iter().map(|d| Contradiction {
                topic: d.topic.clone(),
                views: d.views.iter().map(|v| ContradictionView {
                    model: v.model.clone(),
                    position: v.position.clone(),
                }).collect(),
            }).collect(),
            blind_spots: comparison.coverage_gaps.clone(),
            worker_results,
            judge_output_raw: None,
            total_cost_usd: total_cost,
            models_succeeded: responses.iter().filter(|r| r.error.is_none()).count(),
            models_failed: responses.iter().filter(|r| r.error.is_some()).count(),
            duration_ms,
            timestamp: chrono::Utc::now().to_rfc3339(),
            original_prompt: params.prompt.clone(),
        };
        let _ = engine.sessions().save(&result).await;
    }

    Ok(DiffResult {
        session_id,
        responses,
        comparison,
        cost_usd: total_cost,
        duration_ms,
    })
}

fn build_comparison(responses: &[WorkerResponseView]) -> DiffComparison {
    use imara_diff::{Algorithm, Diff, InternedInput};

    let successful: Vec<&WorkerResponseView> = responses
        .iter()
        .filter(|r| r.error.is_none() && !r.content.is_empty())
        .collect();

    if successful.len() < 2 {
        let unique_insights: Vec<ModelInsight> = successful
            .iter()
            .map(|r| ModelInsight {
                model: r.model.clone(),
                insight: r.content.chars().take(300).collect(),
            })
            .collect();
        return DiffComparison {
            common_points: vec![],
            unique_insights,
            disagreements: vec![],
            coverage_gaps: vec![],
            best_by_aspect: vec![],
        };
    }

    // ── 1. Line-level diffs between every model pair ──
    struct PairAnalysis {
        model_a: String,
        model_b: String,
        common_lines: Vec<String>,
        a_only: Vec<String>,
        b_only: Vec<String>,
    }

    let mut pair_analyses: Vec<PairAnalysis> = Vec::new();
    for i in 0..successful.len() {
        for j in (i + 1)..successful.len() {
            let input = InternedInput::new(
                successful[i].content.as_str(),
                successful[j].content.as_str(),
            );
            let mut diff = Diff::compute(Algorithm::Histogram, &input);
            diff.postprocess_lines(&input);

            let mut common_lines = Vec::new();
            let mut a_only = Vec::new();
            let mut b_only = Vec::new();

            // Lines in "before" that are NOT removed → common
            for (idx, &token) in input.before.iter().enumerate() {
                let line = input.interner[token].trim();
                if line.is_empty() {
                    continue;
                }
                if diff.is_removed(idx as u32) {
                    a_only.push(line.to_string());
                } else {
                    common_lines.push(line.to_string());
                }
            }
            // Lines in "after" that ARE added → b_only
            for (idx, &token) in input.after.iter().enumerate() {
                let line = input.interner[token].trim();
                if line.is_empty() {
                    continue;
                }
                if diff.is_added(idx as u32) {
                    b_only.push(line.to_string());
                }
                // Already got common from before side, no need to re-add
            }

            pair_analyses.push(PairAnalysis {
                model_a: successful[i].model.clone(),
                model_b: successful[j].model.clone(),
                common_lines,
                a_only,
                b_only,
            });
        }
    }

    // ── 2. Common points: lines shared by ALL model pairs ──
    let mut common_points: Vec<String> = Vec::new();
    if let Some(first_pair) = pair_analyses.first() {
        'line_loop: for line in &first_pair.common_lines {
            // Check that all other pairs also have this line in common
            for pair in &pair_analyses[1..] {
                if !pair.common_lines.iter().any(|l| {
                    imara_diff_similarity(l, line) > 0.85
                }) {
                    continue 'line_loop;
                }
            }
            let display = truncate(line, 150);
            if !common_points.contains(&display) {
                common_points.push(display);
            }
        }
    }
    common_points.truncate(10);

    // ── 3. Unique insights: lines only in one model, absent from all others ──
    let mut unique_insights: Vec<ModelInsight> = Vec::new();
    // Collect per-model unique lines from pair analyses
    for resp in &successful {
        let mut model_only: Vec<String> = Vec::new();
        for pair in &pair_analyses {
            if pair.model_a == resp.model {
                model_only.extend(pair.a_only.clone());
            } else if pair.model_b == resp.model {
                model_only.extend(pair.b_only.clone());
            }
        }
        // Pick significant unique lines
        let significant: Vec<String> = model_only
            .into_iter()
            .filter(|l| l.len() > 30)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(3)
            .collect();
        for line in significant {
            unique_insights.push(ModelInsight {
                model: resp.model.clone(),
                insight: truncate(&line, 200),
            });
        }
    }

    // ── 4. Disagreements: modified hunks where models replaced content ──
    let mut disagreements: Vec<Disagreement> = Vec::new();
    for pair in &pair_analyses {
        // When one model has lines the other removed, and vice versa, that's a disagreement
        if !pair.a_only.is_empty() && !pair.b_only.is_empty() {
            let topic = extract_topic(&pair.a_only[0]);
            disagreements.push(Disagreement {
                topic,
                views: vec![
                    DisagreementView {
                        model: pair.model_a.clone(),
                        position: truncate(&pair.a_only[0], 250),
                    },
                    DisagreementView {
                        model: pair.model_b.clone(),
                        position: truncate(&pair.b_only[0], 250),
                    },
                ],
            });
        }
    }
    disagreements.truncate(5);

    // ── 5. Coverage gaps: important themes absent from all responses ──
    let all_content: String = successful
        .iter()
        .map(|r| r.content.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    let question_words = ["why", "how", "risk", "limitation", "caveat", "alternative", "cost", "security", "performance"];
    let mut coverage_gaps: Vec<String> = Vec::new();
    for word in &question_words {
        if !all_content.contains(word) {
            coverage_gaps.push(format!("No model explicitly addressed '{word}' aspects"));
        }
    }

    // ── 6. Best by aspect: use diff stats for ranking ──
    let mut best_by_aspect: Vec<AspectRanking> = Vec::new();
    if !successful.is_empty() {
        let longest = successful.iter().max_by_key(|r| r.content.len()).unwrap();
        best_by_aspect.push(AspectRanking {
            aspect: "thoroughness".into(),
            best_model: longest.model.clone(),
            reason: format!("Most content ({} lines)", longest.content.lines().count()),
        });
        let shortest = successful.iter().min_by_key(|r| r.content.len()).unwrap();
        best_by_aspect.push(AspectRanking {
            aspect: "conciseness".into(),
            best_model: shortest.model.clone(),
            reason: format!("Most concise ({} lines)", shortest.content.lines().count()),
        });
        // Add a "uniqueness" aspect based on diff stats
        let most_unique = successful.iter().max_by_key(|r| {
            pair_analyses.iter()
                .filter(|p| p.model_a == r.model)
                .map(|p| p.a_only.len())
                .chain(pair_analyses.iter().filter(|p| p.model_b == r.model).map(|p| p.b_only.len()))
                .sum::<usize>()
        }).unwrap();
        best_by_aspect.push(AspectRanking {
            aspect: "unique insights".into(),
            best_model: most_unique.model.clone(),
            reason: "Most content not found in other models".into(),
        });
    }

    DiffComparison {
        common_points,
        unique_insights,
        disagreements,
        coverage_gaps,
        best_by_aspect,
    }
}

/// Jaccard-based word similarity using HashSet intersection.
fn imara_diff_similarity(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    intersection as f64 / union as f64
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len.min(s.len())])
    } else {
        s.to_string()
    }
}

fn extract_topic(sent: &str) -> String {
    let topic = sent.split_whitespace().take(5).collect::<Vec<_>>().join(" ");
    truncate(&topic, 60)
}
