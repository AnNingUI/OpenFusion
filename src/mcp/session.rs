//! fusion_session tool: archive management.

use crate::engine::FusionEngine;
use crate::error::OpenFusionError;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
pub struct SessionParams {
    pub action: String,
    pub session_id: Option<String>,
    pub query: Option<String>,
    pub limit: Option<usize>,
    pub export_format: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum SessionResult {
    List(SessionListResult),
    Get(SessionGetResult),
    Export(SessionExportResult),
    Delete(SessionDeleteResult),
    Replay(SessionReplayResult),
}

#[derive(Debug, Serialize)]
pub struct SessionListResult {
    pub action: String,
    pub sessions: Vec<crate::engine::session::SessionMeta>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct SessionGetResult {
    pub action: String,
    pub session: Option<crate::engine::session::FusionSession>,
    pub raw_json: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SessionExportResult {
    pub action: String,
    pub file_path: String,
}

#[derive(Debug, Serialize)]
pub struct SessionDeleteResult {
    pub action: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct SessionReplayResult {
    pub action: String,
    pub session_id: String,
    pub judge_mode: bool,
    pub panel_override: Vec<String>,
    pub fresh_result: Option<crate::engine::FusionResult>,
}

pub async fn run(
    engine: Arc<FusionEngine>,
    params: SessionParams,
) -> Result<SessionResult, OpenFusionError> {
    match params.action.as_str() {
        "list" => {
            let sessions = engine
                .sessions()
                .list(params.query.as_deref(), params.limit.or(Some(20)))
                .await?;
            let total = sessions.len();
            Ok(SessionResult::List(SessionListResult {
                action: "list".into(),
                sessions,
                total,
            }))
        }
        "get" => {
            let id = required_session_id(params.session_id, "get")?;
            let session = engine.sessions().get(&id).await?;
            let raw_json = serde_json::to_string_pretty(&session).ok();
            let is_json = params.export_format.as_deref() == Some("json");
            Ok(SessionResult::Get(SessionGetResult {
                action: "get".into(),
                session: if is_json { None } else { Some(session) },
                raw_json,
            }))
        }
        "export" => {
            let id = required_session_id(params.session_id, "export")?;
            let path = engine.sessions().export_markdown(&id).await?;
            Ok(SessionResult::Export(SessionExportResult {
                action: "export".into(),
                file_path: path.display().to_string(),
            }))
        }
        "delete" => {
            let id = required_session_id(params.session_id, "delete")?;
            let deleted = engine.sessions().delete(&id).await?;
            Ok(SessionResult::Delete(SessionDeleteResult {
                action: "delete".into(),
                deleted,
            }))
        }
        "replay" => {
            let id = required_session_id(params.session_id, "replay")?;
            let session = engine.sessions().get(&id).await?;
            let panel_override = session
                .full
                .worker_results
                .iter()
                .map(|worker| {
                    if worker.name.is_empty() {
                        worker.model.clone()
                    } else {
                        worker.name.clone()
                    }
                })
                .collect::<Vec<_>>();
            let judge_mode = session.full.enhanced_prompt_used;

            let request = crate::protocol::IntermediateRequest {
                messages: vec![crate::protocol::Message {
                    role: crate::protocol::Role::User,
                    content: session.meta.original_prompt.clone(),
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
            };

            let fresh = engine
                .execute_advisory(&request, &panel_override, judge_mode, true)
                .await?;
            Ok(SessionResult::Replay(SessionReplayResult {
                action: "replay".into(),
                session_id: id,
                judge_mode,
                panel_override,
                fresh_result: Some(fresh),
            }))
        }
        _ => Err(OpenFusionError::Config(format!(
            "Unknown action '{}'. Valid: list, get, export, delete, replay",
            params.action
        ))),
    }
}

fn required_session_id(id: Option<String>, action: &str) -> Result<String, OpenFusionError> {
    id.ok_or_else(|| OpenFusionError::Config(format!("session_id required for {action}")))
}
