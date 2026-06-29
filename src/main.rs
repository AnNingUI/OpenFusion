//! OpenFusion — MCP multi-model advisory panel.
//!
//! Usage:
//!   openfusion [--config PATH]
//!
//! Config defaults to ~/.openfusion/config.toml

mod config;
mod engine;
mod error;
mod mcp;
mod protocol;
mod worker;

use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let config_path = parse_arg(&args, "--config");

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

    tracing::info!("Starting MCP server over stdio...");
    crate::mcp::serve(engine).await?;

    Ok(())
}

fn parse_arg(args: &[String], flag: &str) -> Option<PathBuf> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}
