//! Session persistence: save, list, read, delete fusion results.
//!
//! Sessions are stored as JSONL files in the sessions directory.
//! An index file (`_index.jsonl`) maintains lightweight metadata for fast listing.

use crate::engine::FusionResult;
use crate::error::OpenFusionError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub prompt_preview: String,
    pub original_prompt: String,
    pub models: Vec<String>,
    pub models_succeeded: usize,
    pub models_failed: usize,
    pub duration_ms: u64,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionSession {
    pub meta: SessionMeta,
    pub full: FusionResult,
}

pub struct SessionStore {
    dir: PathBuf,
    max_sessions: usize,
    lock: Mutex<()>,
}

impl SessionStore {
    pub fn new(dir: PathBuf, max_sessions: usize) -> Result<Self, OpenFusionError> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            max_sessions,
            lock: Mutex::new(()),
        })
    }

    /// Save a fusion result as a new session.
    pub async fn save(&self, result: &FusionResult) -> Result<(), OpenFusionError> {
        let _guard = self.lock.lock().await;

        let prompt_preview = result.original_prompt.chars().take(200).collect::<String>();
        let original_prompt = result.original_prompt.clone();

        let meta = SessionMeta {
            id: result.session_id.clone(),
            prompt_preview,
            original_prompt,
            models: result.worker_results.iter().map(|w| w.model.clone()).collect(),
            models_succeeded: result.models_succeeded,
            models_failed: result.models_failed,
            duration_ms: result.duration_ms,
            timestamp: result.timestamp.clone(),
        };

        let session = FusionSession {
            meta: meta.clone(),
            full: result.clone(),
        };

        // Write session file
        let session_path = self.dir.join(format!("{}.json", result.session_id));
        let json = serde_json::to_string_pretty(&session)?;
        tokio::fs::write(&session_path, json).await?;

        // Append to index
        let index_path = self.dir.join("_index.jsonl");
        let mut index_line = serde_json::to_string(&meta)?;
        index_line.push('\n');
        let mut f = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&index_path)
            .await?;
        f.write_all(index_line.as_bytes()).await?;

        // Prune old sessions if over max
        self.prune_if_needed().await?;

        Ok(())
    }

    /// List all sessions, newest first.
    pub async fn list(&self, query: Option<&str>, limit: Option<usize>) -> Result<Vec<SessionMeta>, OpenFusionError> {
        let index_path = self.dir.join("_index.jsonl");
        if !index_path.exists() {
            return Ok(vec![]);
        }

        let content = tokio::fs::read_to_string(&index_path).await.unwrap_or_default();
        let mut metas: Vec<SessionMeta> = content
            .lines()
            .filter(|l| !l.is_empty())
            .filter_map(|l| serde_json::from_str::<SessionMeta>(l).ok())
            .collect();

        // Filter by query if provided
        if let Some(q) = query {
            let q = q.to_lowercase();
            metas.retain(|m| {
                m.prompt_preview.to_lowercase().contains(&q)
                    || m.models.iter().any(|model| model.to_lowercase().contains(&q))
            });
        }

        // Newest first (reverse the JSONL order)
        metas.reverse();

        let limit = limit.unwrap_or(50);
        if metas.len() > limit {
            metas.truncate(limit);
        }

        Ok(metas)
    }

    /// Get a full session by ID.
    pub async fn get(&self, session_id: &str) -> Result<FusionSession, OpenFusionError> {
        let session_path = self.dir.join(format!("{session_id}.json"));
        if !session_path.exists() {
            return Err(OpenFusionError::SessionNotFound {
                id: session_id.to_string(),
            });
        }
        let content = tokio::fs::read_to_string(&session_path).await?;
        let session: FusionSession = serde_json::from_str(&content)?;
        Ok(session)
    }

    /// Delete a session by ID.
    pub async fn delete(&self, session_id: &str) -> Result<bool, OpenFusionError> {
        let session_path = self.dir.join(format!("{session_id}.json"));
        if session_path.exists() {
            tokio::fs::remove_file(&session_path).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Export a session as markdown.
    pub async fn export_markdown(&self, session_id: &str) -> Result<PathBuf, OpenFusionError> {
        let session = self.get(session_id).await?;
        let md = session_to_markdown(&session);
        let export_path = self.dir.join(format!("{session_id}.md"));
        tokio::fs::write(&export_path, md).await?;
        Ok(export_path)
    }

    async fn prune_if_needed(&self) -> Result<(), OpenFusionError> {
        let mut metas = self.list(None, None).await?;
        if metas.len() <= self.max_sessions {
            return Ok(());
        }

        // Sort oldest first
        metas.reverse();
        let to_remove = metas.len() - self.max_sessions;

        for meta in metas.iter().take(to_remove) {
            let path = self.dir.join(format!("{}.json", meta.id));
            let _ = tokio::fs::remove_file(&path).await;
        }

        Ok(())
    }
}

fn session_to_markdown(session: &FusionSession) -> String {
    let mut md = String::new();
    md.push_str(&format!("# Fusion Session: {}\n\n", session.meta.id));
    md.push_str(&format!("**Time**: {}\n", session.meta.timestamp));
    md.push_str(&format!("**Duration**: {}ms\n", session.meta.duration_ms));
    md.push_str(&format!(
        "**Models**: {} ({}/{})\n\n",
        session.meta.models.join(", "),
        session.meta.models_succeeded,
        session.meta.models_succeeded + session.meta.models_failed
    ));

    if let Some(ref synthesis) = session.full.synthesis {
        md.push_str("## Synthesis\n\n");
        md.push_str(synthesis);
        md.push_str("\n\n");
    }

    if !session.full.consensus.is_empty() {
        md.push_str("## Consensus\n\n");
        for point in &session.full.consensus {
            md.push_str(&format!("- {point}\n"));
        }
        md.push('\n');
    }

    md.push_str("## Individual Responses\n\n");
    for wr in &session.full.worker_results {
        md.push_str(&format!("### {}\n\n", wr.model));
        if let Some(ref resp) = wr.response {
            md.push_str(&resp.content);
            md.push_str("\n\n");
        } else if let Some(ref err) = wr.error {
            md.push_str(&format!("*Error: {err}*\n\n"));
        }
    }

    md
}
