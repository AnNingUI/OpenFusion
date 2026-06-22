//! OpenFusion — Multi-model fusion gateway with HTTP + MCP dual servers.
//!
//! Usage:
//!   openfusion [--config PATH] [--mcp]
//!
//! If --mcp is passed, runs MCP server only (over stdio).
//! Otherwise, runs HTTP server (and MCP in background if config enables it).
//! Config defaults to ~/.openfusion/config.toml

mod config;
mod engine;
mod error;
mod mcp;
mod protocol;
mod server;
mod worker;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let config_path = parse_arg(&args, "--config");
    let mcp_only = args.iter().any(|a| a == "--mcp");

    // Init tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "openfusion=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    // Load config
    let cfg = config::Config::load(config_path.as_ref())?;
    tracing::info!(
        "OpenFusion v{} starting — {} workers, judge: {}",
        env!("CARGO_PKG_VERSION"),
        cfg.workers.len(),
        cfg.judge.model
    );

    let config = Arc::new(cfg);
    let engine = Arc::new(engine::FusionEngine::new(config.clone())?);

    if mcp_only {
        // MCP-only mode (for Claude Code integration)
        tracing::info!("Starting MCP server over stdio...");
        crate::mcp::serve(engine, config).await?;
    } else {
        // HTTP + MCP dual mode
        let state = server::AppState {
            engine: engine.clone(),
            config: config.clone(),
            start_time: Instant::now(),
            request_count: Arc::new(AtomicU64::new(0)),
        };

        let router = server::build_router(state);
        let bind_addr = format!("{}:{}", config.server.host, config.server.port);
        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

        tracing::info!("HTTP server listening on http://{bind_addr}");
        tracing::info!("  OpenAI Completions:   POST http://{bind_addr}/v1/chat/completions");
        tracing::info!("  OpenAI Responses:     POST http://{bind_addr}/v1/responses");
        tracing::info!("  Google GenAI:         POST http://{bind_addr}/v1/google/models/{{model}}:generateContent");
        tracing::info!("  Anthropic Messages:   POST http://{bind_addr}/v1/messages");
        tracing::info!("  Health:               GET  http://{bind_addr}/health");
        tracing::info!("  Metrics:              GET  http://{bind_addr}/metrics");

        axum::serve(listener, router).await?;
    }

    Ok(())
}

fn parse_arg(args: &[String], flag: &str) -> Option<PathBuf> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}
