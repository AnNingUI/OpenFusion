//! MCP server for OpenFusion integration.

use crate::engine::FusionEngine;
use rmcp::handler::server::ServerHandler;
use rmcp::model::ErrorData as McpError;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerInfo, Tool, ToolsCapability,
};
use rmcp::service::{RequestContext, RoleServer};
use std::sync::Arc;

mod bench;
mod diff;
mod fusion;
mod session;

/// Build and serve the MCP server over stdio.
pub async fn serve(engine: Arc<FusionEngine>) -> Result<(), Box<dyn std::error::Error>> {
    let server = OpenFusionServer::new(engine);
    let running = rmcp::serve_server(server, rmcp::transport::io::stdio()).await?;
    running.waiting().await?;
    Ok(())
}

#[derive(Clone)]
pub struct OpenFusionServer {
    engine: Arc<FusionEngine>,
}

impl OpenFusionServer {
    pub fn new(engine: Arc<FusionEngine>) -> Self {
        Self { engine }
    }
}

impl ServerHandler for OpenFusionServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities.tools = Some(ToolsCapability {
            list_changed: Some(false),
        });
        info.server_info = Implementation::new("openfusion", env!("CARGO_PKG_VERSION"));
        info.instructions = Some(
            "OpenFusion is an MCP-first multi-model advisory panel. Use fusion for normal work: first abstract the concrete repository or file-tree problem into a self-contained, project-independent prompt, then send that prompt to OpenFusion workers. Do not pass raw repo dumps, local file trees, or client tool instructions. judge_mode=false returns raw worker opinions for the calling AI client to synthesize and implement; judge_mode=true lets OpenFusion run its configured judge and return a concise synthesis. fusion_diff returns raw worker outputs only; fusion_bench reports measurable success, latency, and cost only."
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

        let content = match request.name.as_ref() {
            "fusion" => {
                let params =
                    serde_json::from_value::<fusion::FusionParams>(serde_json::Value::Object(args))
                        .map_err(|e| {
                            McpError::invalid_params(format!("Invalid params: {e}"), None)
                        })?;
                let result = fusion::run(self.engine.clone(), params)
                    .await
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
                serialize_tool_result(&result)
            }
            "fusion_diff" => {
                let params =
                    serde_json::from_value::<diff::DiffParams>(serde_json::Value::Object(args))
                        .map_err(|e| {
                            McpError::invalid_params(format!("Invalid params: {e}"), None)
                        })?;
                let result = diff::run(self.engine.clone(), params)
                    .await
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
                serialize_tool_result(&result)
            }
            "fusion_bench" => {
                let params = serde_json::from_value::<bench::BenchParams>(
                    serde_json::Value::Object(args),
                )
                .map_err(|e| McpError::invalid_params(format!("Invalid params: {e}"), None))?;
                let result = bench::run(self.engine.clone(), params)
                    .await
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
                serialize_tool_result(&result)
            }
            "fusion_session" => {
                let params = serde_json::from_value::<session::SessionParams>(
                    serde_json::Value::Object(args),
                )
                .map_err(|e| McpError::invalid_params(format!("Invalid params: {e}"), None))?;
                let result = session::run(self.engine.clone(), params)
                    .await
                    .map_err(|e| McpError::invalid_params(e.to_string(), None))?;
                serialize_tool_result(&result)
            }
            _ => {
                return Err(McpError::invalid_params(
                    format!("Unknown tool: {}", request.name),
                    None,
                ));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(content)]))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: vec![
                Tool::new(
                    "fusion",
                    "Primary OpenFusion tool. Send an abstract, self-contained, project-independent problem statement to multiple advisory workers. judge_mode=false returns raw worker outputs; judge_mode=true runs OpenFusion's configured judge.",
                    json_to_schema(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Required. A self-contained abstract problem statement. Do not include raw file trees, large repo dumps, local paths as required context, or AI-client tool instructions."
                            },
                            "system": {
                                "type": "string",
                                "description": "Optional project-independent domain framing, prepended to the worker-visible prompt."
                            },
                            "judge_mode": {
                                "type": "boolean",
                                "description": "false: return raw worker outputs for the calling AI client to synthesize. true: run OpenFusion's judge and return a concise synthesis."
                            },
                            "panel_override": {
                                "type": "array",
                                "description": "Optional worker names or model ids to use instead of all configured workers.",
                                "items": { "type": "string" }
                            },
                            "save_session": {
                                "type": "boolean",
                                "description": "Whether to archive this run."
                            }
                        },
                        "required": ["prompt"]
                    })),
                ),
                Tool::new(
                    "fusion_diff",
                    "Send a prompt to workers and return raw model outputs plus measurable metadata only. No semantic consensus, disagreement, or quality inference is performed.",
                    json_to_schema(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string", "description": "The abstract prompt to analyze." },
                            "system": { "type": "string", "description": "Optional project-independent domain framing, prepended to the worker-visible prompt." },
                            "panel_override": {
                                "type": "array",
                                "description": "Optional worker names or model ids to use instead of all configured workers.",
                                "items": { "type": "string" }
                            },
                            "save_session": { "type": "boolean", "description": "Whether to archive this run." }
                        },
                        "required": ["prompt"]
                    })),
                ),
                Tool::new(
                    "fusion_bench",
                    "Run the same prompt through multiple fusion profiles, returning measurable success, latency, and reported cost. It does not claim to score answer quality.",
                    json_to_schema(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string", "description": "The abstract prompt to benchmark." },
                            "system": { "type": "string", "description": "Optional project-independent domain framing, prepended to the prompt." },
                            "runs": { "type": "integer", "description": "Runs per config. Defaults to 1." },
                            "configs": {
                                "type": "array",
                                "description": "Fusion profiles to test.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": { "type": "string" },
                                        "workers": {
                                            "type": "array",
                                            "description": "Worker names or model ids. Empty or omitted means all configured workers.",
                                            "items": { "type": "string" }
                                        },
                                        "judge_mode": {
                                            "type": "boolean",
                                            "description": "Whether this profile includes judge synthesis."
                                        }
                                    },
                                    "required": ["label"]
                                }
                            }
                        },
                        "required": ["prompt", "configs"]
                    })),
                ),
                Tool::new(
                    "fusion_session",
                    "Manage fusion session archives. Actions: list, get, export, delete, replay. Replay reruns the archived prompt with the original judge_mode and archived worker panel when possible.",
                    json_to_schema(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "action": { "type": "string", "description": "list | get | export | delete | replay" },
                            "session_id": { "type": "string", "description": "Required for get/export/delete/replay." },
                            "query": { "type": "string", "description": "Search query for list action." },
                            "limit": { "type": "integer", "description": "Max results for list. Defaults to 20." },
                            "export_format": { "type": "string", "description": "json or markdown. Defaults to markdown." }
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

fn serialize_tool_result<T: serde::Serialize>(result: &T) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|e| format!("{{ \"error\": \"{e}\" }}"))
}

fn json_to_schema(v: serde_json::Value) -> Arc<serde_json::Map<String, serde_json::Value>> {
    Arc::new(v.as_object().cloned().unwrap_or_default())
}
