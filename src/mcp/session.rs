//! fusion_session tool: archive management.

use crate::engine::FusionEngine;
use crate::error::OpenFusionError;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize, Default)]
pub struct SessionParams {
    /// Action: list, get, export, delete, replay
    pub action: String,
    pub session_id: Option<String>,
    /// Search query for list action
    pub query: Option<String>,
    /// Max results for list (default 20)
    pub limit: Option<usize>,
    /// Export format: "json" or "markdown" (default markdown)
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
    pub fresh_result: Option<crate::engine::FusionResult>,
}

pub async fn run(engine: Arc<FusionEngine>, params: SessionParams) -> Result<SessionResult, OpenFusionError> {
    match params.action.as_str() {
        "list" => {
            let sessions = engine.sessions().list(
                params.query.as_deref(),
                params.limit.or(Some(20)),
            ).await?;
            let total = sessions.len();
            Ok(SessionResult::List(SessionListResult {
                action: "list".into(),
                sessions,
                total,
            }))
        }

        "get" => {
            let id = params.session_id.ok_or_else(|| {
                OpenFusionError::Config("session_id required for get".into())
            })?;
            let session = engine.sessions().get(&id).await?;
            let raw_json = serde_json::to_string_pretty(&session).ok();
            let is_json = params.export_format.as_deref() == Some("json");
            Ok(SessionResult::Get(SessionGetResult {
                action: "get".into(),
                // For JSON format, suppress the structured session (caller reads raw_json)
                session: if is_json { None } else { Some(session) },
                raw_json,
            }))
        }

        "export" => {
            let id = params.session_id.ok_or_else(|| {
                OpenFusionError::Config("session_id required for export".into())
            })?;
            let path = engine.sessions().export_markdown(&id).await?;
            Ok(SessionResult::Export(SessionExportResult {
                action: "export".into(),
                file_path: path.display().to_string(),
            }))
        }

        "delete" => {
            let id = params.session_id.ok_or_else(|| {
                OpenFusionError::Config("session_id required for delete".into())
            })?;
            let deleted = engine.sessions().delete(&id).await?;
            Ok(SessionResult::Delete(SessionDeleteResult {
                action: "delete".into(),
                deleted,
            }))
        }

        "replay" => {
            let id = params.session_id.ok_or_else(|| {
                OpenFusionError::Config("session_id required for replay".into())
            })?;
            // Load original session to get the full prompt
            let session = engine.sessions().get(&id).await?;
            // Use the stored original_prompt for replay — not the preview
            let user_text = session.meta.original_prompt.clone();
            let messages = vec![
                crate::protocol::Message {
                    role: crate::protocol::Role::User,
                    content: user_text,
                },
            ];
            let request = crate::protocol::IntermediateRequest {
                messages,
                max_tokens: 2048,
                temperature: Some(0.7),
                stop: vec![],
                api_key: None,
            };
            let fresh = engine.execute(&request).await?;
            Ok(SessionResult::Replay(SessionReplayResult {
                action: "replay".into(),
                session_id: id,
                fresh_result: Some(fresh),
            }))
        }

        _ => Err(OpenFusionError::Config(format!(
            "Unknown action '{}'. Valid: list, get, export, delete, replay",
            params.action
        ))),
    }
}
