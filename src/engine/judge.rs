//! Judge prompt builder and structured output parser.

use crate::engine::Contradiction;
use crate::protocol::{IntermediateRequest, WorkerResult};

#[derive(Debug, Clone)]
pub struct JudgeOutput {
    pub synthesis: String,
    pub consensus: Vec<String>,
    pub contradictions: Vec<Contradiction>,
    pub blind_spots: Vec<String>,
}

/// Build the judge prompt from worker responses.
pub fn build_judge_prompt(
    request: &IntermediateRequest,
    worker_results: &[&WorkerResult],
) -> String {
    let user_query = request
        .messages
        .iter()
        .filter(|m| matches!(m.role, crate::protocol::Role::User))
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let system_context = request
        .messages
        .iter()
        .filter(|m| matches!(m.role, crate::protocol::Role::System))
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let mut worker_answers = String::new();
    for (i, wr) in worker_results.iter().enumerate() {
        if let Some(resp) = &wr.response {
            worker_answers.push_str(&format!(
                "### Model {}: {}\n\n{}\n\n",
                i + 1,
                wr.model,
                resp.content
            ));
        }
    }

    format!(
        r#"You are a synthesis judge. Analyze the following responses from multiple AI models to the same question, then produce a structured analysis.

## User Question
{user_query}

## System Context (if any)
{system_context}

## Model Responses
{worker_answers}

## Instructions

Analyze all responses and output the following sections using EXACTLY the markers shown:

### CONSENSUS
List all points where the models agree. One bullet per point.
- Point 1
- Point 2

### CONTRADICTIONS
List topics where models disagree. For each, note which model said what.
- **Topic**: description
  - Model A: its position
  - Model B: its position

### BLIND SPOTS
List important angles, risks, or caveats that NO model addressed.
- Blind spot 1
- Blind spot 2

### SYNTHESIS
Write the best possible unified answer, combining the strongest reasoning from all models. This should read as a single polished response. Do NOT mention individual model names in the synthesis.
"#
    )
}

/// Parse the judge's structured output. Loose parser — missing sections are
/// returned empty rather than failing.
pub fn parse_judge_output(raw: &str) -> Result<JudgeOutput, crate::error::OpenFusionError> {
    let synthesis = extract_section(raw, "SYNTHESIS");
    let consensus_raw = extract_section(raw, "CONSENSUS");
    let contradictions_raw = extract_section(raw, "CONTRADICTIONS");
    let blind_spots_raw = extract_section(raw, "BLIND SPOTS");

    let consensus = parse_bullets(&consensus_raw);
    let contradictions = parse_contradictions(&contradictions_raw);
    let blind_spots = parse_bullets(&blind_spots_raw);

    // If synthesis is empty, use the full raw text as fallback
    let synthesis = if synthesis.trim().is_empty() {
        raw.to_string()
    } else {
        synthesis
    };

    Ok(JudgeOutput {
        synthesis,
        consensus,
        contradictions,
        blind_spots,
    })
}

fn extract_section(text: &str, marker: &str) -> String {
    let pattern = format!("### {marker}");
    let start = text.find(&pattern);

    if let Some(start_idx) = start {
        let section_start = start_idx + pattern.len();
        let remainder = &text[section_start..];

        // Find next "### " marker
        if let Some(next_marker) = remainder.find("\n### ") {
            remainder[..next_marker].trim().to_string()
        } else {
            remainder.trim().to_string()
        }
    } else {
        // Try without the ### prefix
        let alt_pattern = marker.to_string();
        if let Some(start_idx) = text.find(&alt_pattern) {
            let section_start = start_idx + alt_pattern.len();
            let remainder = &text[section_start..];
            if let Some(next_marker) = remainder.find("\n### ") {
                remainder[..next_marker].trim().to_string()
            } else {
                remainder.trim().to_string()
            }
        } else {
            String::new()
        }
    }
}

fn parse_bullets(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| line.trim_start().starts_with('-'))
        .map(|line| line.trim_start_matches('-').trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_contradictions(text: &str) -> Vec<Contradiction> {
    let mut contradictions = Vec::new();
    let mut current_topic: Option<String> = None;
    let mut current_views: Vec<crate::engine::ContradictionView> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("- **") || trimmed.starts_with("- **") {
            // Save previous
            if let Some(topic) = current_topic.take() {
                contradictions.push(Contradiction {
                    topic,
                    views: std::mem::take(&mut current_views),
                });
            }
            // Extract topic
            if let Some(_topic_end) = trimmed.find("**") {
                let rest = &trimmed[2..]; // skip "- "
                if let Some(topic_start) = rest.find("**") {
                    let topic = &rest[topic_start + 2..];
                    if let Some(colon) = topic.find(':') {
                        current_topic = Some(topic[..colon].trim().to_string());
                    } else {
                        current_topic = Some(topic.trim().to_string());
                    }
                }
            }
        } else if trimmed.starts_with("  - ") && current_topic.is_some() {
            let content = trimmed[4..].trim();
            if let Some(colon) = content.find(':') {
                let model = content[..colon].trim().to_string();
                let position = content[colon + 1..].trim().to_string();
                current_views.push(crate::engine::ContradictionView { model, position });
            }
        }
    }

    // Save last
    if let Some(topic) = current_topic
        && !topic.is_empty() {
            contradictions.push(Contradiction {
                topic,
                views: std::mem::take(&mut current_views),
            });
        }

    contradictions
}
