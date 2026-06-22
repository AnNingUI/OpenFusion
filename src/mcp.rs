//! MCP server for Claude Code integration.
//!
//! Implements ServerHandler manually (avoids tool_router macro and schemars version issues).
//! Three tools: fusion_diff, fusion_bench, fusion_session.

use crate::config::Config;
use crate::engine::FusionEngine;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerInfo, Tool, ToolsCapability,
};
use rmcp::model::ErrorData as McpError;
use rmcp::service::{RequestContext, RoleServer};
use std::sync::Arc;

mod bench;
mod diff;
mod session;

/// Build and serve the MCP server over stdio.
pub async fn serve(
    engine: Arc<FusionEngine>,
    config: Arc<Config>,
) -> Result<(), Box<dyn std::error::Error>> {
    let server = OpenFusionServer::new(engine, config);
    let running = rmcp::serve_server(server, rmcp::transport::io::stdio()).await?;
    running.waiting().await?;
    Ok(())
}

#[derive(Clone)]
pub struct OpenFusionServer {
    engine: Arc<FusionEngine>,
    config: Arc<Config>,
}

impl OpenFusionServer {
    pub fn new(engine: Arc<FusionEngine>, config: Arc<Config>) -> Self {
        Self { engine, config }
    }
}

// ── Manual ServerHandler implementation ──

impl ServerHandler for OpenFusionServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities.tools = Some(ToolsCapability {
            list_changed: Some(false),
        });
        info.server_info = Implementation::new(
            "openfusion",
            env!("CARGO_PKG_VERSION"),
        );
        info.instructions = Some(
            "OpenFusion — multi-model fusion gateway. Tools: fusion_diff (model comparison), fusion_bench (config benchmarking), fusion_session (session archive management)."
                .into(),
        );
        info
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let args = request.arguments.unwrap_or_default();

        match request.name.as_ref() {
            "fusion_diff" => {
                let p = serde_json::from_value::<diff::DiffParams>(serde_json::Value::Object(args))
                    .map_err(|e| McpError::invalid_params(
                        format!("Invalid params: {e}"), None,
                    ))?;
                let result = diff::run(self.engine.clone(), p)
                    .await
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
                let content = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|e| format!("{{ \"error\": \"{e}\" }}"));
                Ok(CallToolResult::success(vec![Content::text(content)]))
            }
            "fusion_bench" => {
                let p = serde_json::from_value::<bench::BenchParams>(serde_json::Value::Object(args))
                    .map_err(|e| McpError::invalid_params(
                        format!("Invalid params: {e}"), None,
                    ))?;
                let result = bench::run(self.engine.clone(), self.config.clone(), p)
                    .await
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
                let content = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|e| format!("{{ \"error\": \"{e}\" }}"));
                Ok(CallToolResult::success(vec![Content::text(content)]))
            }
            "fusion_session" => {
                let p = serde_json::from_value::<session::SessionParams>(serde_json::Value::Object(args))
                    .map_err(|e| McpError::invalid_params(
                        format!("Invalid params: {e}"), None,
                    ))?;
                let result = session::run(self.engine.clone(), p)
                    .await
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
                let content = serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|e| format!("{{ \"error\": \"{e}\" }}"));
                Ok(CallToolResult::success(vec![Content::text(content)]))
            }
            _ => Err(McpError::invalid_params(
                format!("Unknown tool: {}", request.name),
                None,
            )),
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: vec![
                Tool::new(
                    "fusion_diff",
                    "Send a prompt to all workers in parallel, return each model's raw response plus a structured comparison matrix (unique insights, disagreements, coverage gaps). No synthesis step — Claude Code acts as the judge.",
                    json_to_schema(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string", "description": "The question/prompt to analyze" },
                            "system": { "type": "string", "description": "Optional system context" },
                            "panel_override": { "type": "string", "description": "Comma-separated model names to use instead of defaults" },
                            "save_session": { "type": "boolean", "description": "Whether to archive this run" }
                        },
                        "required": ["prompt"]
                    })),
                ),
                Tool::new(
                    "fusion_bench",
                    "Run the same prompt through multiple configuration profiles, returning a cost/quality/latency comparison matrix.",
                    json_to_schema(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string", "description": "The prompt to benchmark" },
                            "runs": { "type": "integer", "description": "Runs per config (default 1)" },
                            "configs": {
                                "type": "array",
                                "description": "Config profiles to test",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": { "type": "string" },
                                        "workers": { "type": "string", "description": "Comma-separated model names" },
                                        "judge": { "type": "string", "description": "Judge model (optional)" }
                                    }
                                }
                            }
                        },
                        "required": ["prompt", "configs"]
                    })),
                ),
                Tool::new(
                    "fusion_session",
                    "Manage fusion session archives. Actions: list, get, export (markdown), delete, replay.",
                    json_to_schema(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "action": { "type": "string", "description": "list | get | export | delete | replay" },
                            "session_id": { "type": "string", "description": "Required for get/export/delete/replay" },
                            "query": { "type": "string", "description": "Search query for list action" },
                            "limit": { "type": "integer", "description": "Max results for list (default 20)" },
                            "export_format": { "type": "string", "description": "json or markdown (default markdown)" }
                        },
                        "required": ["action"]
                    })),
                ),
            ],
            next_cursor: None,
            meta: None,
        })
    }
}

/// Convert a serde_json::Value object to the Arc<JsonObject> that Tool input_schema expects.
fn json_to_schema(v: serde_json::Value) -> Arc<serde_json::Map<String, serde_json::Value>> {
    Arc::new(v.as_object().cloned().unwrap_or_default())
}
