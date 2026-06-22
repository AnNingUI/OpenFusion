use crate::error::OpenFusionError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Worker capability flags (all on by default).
fn default_true() -> bool { true }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,

    #[serde(default)]
    pub fusion: FusionConfig,

    pub judge: ModelEndpoint,

    #[serde(default)]
    pub workers: Vec<ModelEndpoint>,

    #[serde(default)]
    pub storage: StorageConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FusionConfig {
    #[serde(default = "default_fusion_name")]
    pub name: String,
    #[serde(default = "default_fusion_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_worker_timeout")]
    pub worker_timeout_secs: u64,
    #[serde(default = "default_min_workers")]
    pub min_workers: usize,
    #[serde(default)]
    pub retry: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelEndpoint {
    pub base_url: String,
    pub api: ApiProtocol,
    pub api_key: String,
    pub model: String,
    #[serde(default = "default_true")]
    pub web_search: bool,
    #[serde(default = "default_true")]
    pub web_fetch: bool,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ApiProtocol {
    OpenaiCompletions,
    OpenaiResponses,
    GoogleGenerativeAi,
    AnthropicMessages,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StorageConfig {
    #[serde(default = "default_sessions_dir")]
    pub sessions_dir: PathBuf,
    #[serde(default = "default_max_sessions")]
    pub max_sessions: usize,
}

// ── defaults ──

fn default_port() -> u16 { 9999 }
fn default_host() -> String { "127.0.0.1".into() }
fn default_max_concurrent() -> usize { 10 }
fn default_log_level() -> String { "info".into() }
fn default_fusion_name() -> String { "openrouter/fusion".into() }
fn default_fusion_timeout() -> u64 { 120 }
fn default_worker_timeout() -> u64 { 60 }
fn default_min_workers() -> usize { 1 }
fn default_sessions_dir() -> PathBuf { dirs_home().join(".openfusion").join("sessions") }
fn default_max_sessions() -> usize { 1000 }

fn dirs_home() -> PathBuf {
    directories::UserDirs::new()
        .map(|d| d.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            host: default_host(),
            max_concurrent_requests: default_max_concurrent(),
            log_level: default_log_level(),
        }
    }
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            name: default_fusion_name(),
            timeout_secs: default_fusion_timeout(),
            worker_timeout_secs: default_worker_timeout(),
            min_workers: default_min_workers(),
            retry: 0,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            sessions_dir: default_sessions_dir(),
            max_sessions: default_max_sessions(),
        }
    }
}

impl Config {
    /// Load config from `~/.openfusion/config.toml`. Creates default if missing.
    pub fn load(path: Option<&PathBuf>) -> Result<Self, OpenFusionError> {
        let config_path = path
            .cloned()
            .unwrap_or_else(|| dirs_home().join(".openfusion").join("config.toml"));

        if !config_path.exists() {
            let default_config = Config::default_config();
            if let Some(parent) = config_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let toml_str = toml::to_string_pretty(&default_config)
                .map_err(|e| OpenFusionError::Config(format!("Failed to serialize default config: {e}")))?;
            std::fs::write(&config_path, toml_str)?;
            tracing::info!("Created default config at {}", config_path.display());
            return Ok(default_config);
        }

        let content = std::fs::read_to_string(&config_path)?;
        let config: Config = toml::from_str(&content)
            .map_err(|e| OpenFusionError::Config(format!("Failed to parse config: {e}")))?;
        config.validate()?;
        tracing::info!("Loaded config from {}", config_path.display());
        Ok(config)
    }

    fn default_config() -> Self {
        Self {
            server: ServerConfig::default(),
            fusion: FusionConfig::default(),
            judge: ModelEndpoint {
                base_url: "https://api.openai.com".into(),
                api: ApiProtocol::OpenaiCompletions,
                api_key: "sk-xxx".into(),
                model: "deepseek-v4-pro".into(),
                web_search: true,
                web_fetch: true,
                system_prompt: None,
            },
            workers: vec![
                ModelEndpoint {
                    base_url: "https://api.openai.com".into(),
                    api: ApiProtocol::OpenaiResponses,
                    api_key: "sk-xxx".into(),
                    model: "gpt-5.5".into(),
                    web_search: true,
                    web_fetch: true,
                    system_prompt: None,
                },
                ModelEndpoint {
                    base_url: "https://generativelanguage.googleapis.com".into(),
                    api: ApiProtocol::GoogleGenerativeAi,
                    api_key: "sk-xxx".into(),
                    model: "gemini-3.5-flash".into(),
                    web_search: true,
                    web_fetch: true,
                    system_prompt: None,
                },
                ModelEndpoint {
                    base_url: "https://api.anthropic.com".into(),
                    api: ApiProtocol::AnthropicMessages,
                    api_key: "sk-xxx".into(),
                    model: "claude-sonnet-4.6".into(),
                    web_search: true,
                    web_fetch: true,
                    system_prompt: None,
                },
            ],
            storage: StorageConfig::default(),
        }
    }

    fn validate(&self) -> Result<(), OpenFusionError> {
        if self.workers.is_empty() {
            return Err(OpenFusionError::Config("At least one worker is required".into()));
        }
        if self.fusion.min_workers > self.workers.len() {
            return Err(OpenFusionError::Config(format!(
                "min_workers ({}) exceeds total workers ({})",
                self.fusion.min_workers,
                self.workers.len()
            )));
        }
        // Validate server port range
        if self.server.port == 0 {
            return Err(OpenFusionError::Config(format!(
                "Invalid port: {}. Must be 1..=65535",
                self.server.port
            )));
        }
        // Validate timeouts
        if self.fusion.timeout_secs == 0 {
            return Err(OpenFusionError::Config(
                "fusion.timeout_secs must be > 0".into(),
            ));
        }
        if self.fusion.worker_timeout_secs == 0 {
            return Err(OpenFusionError::Config(
                "fusion.worker_timeout_secs must be > 0".into(),
            ));
        }
        // Validate base_url scheme
        for (idx, worker) in self.workers.iter().enumerate() {
            if !worker.base_url.starts_with("http://") && !worker.base_url.starts_with("https://") {
                return Err(OpenFusionError::Config(format!(
                    "worker[{}] base_url must start with http:// or https://: {}",
                    idx, worker.base_url
                )));
            }
            if worker.api_key == "sk-xxx" {
                tracing::warn!(
                    "worker[{}] ({}) api_key is the placeholder 'sk-xxx' — set OPENFUSION_API_KEY env var or real key in config",
                    idx,
                    worker.model
                );
            }
        }
        if !self.judge.base_url.starts_with("http://") && !self.judge.base_url.starts_with("https://") {
            return Err(OpenFusionError::Config(format!(
                "judge base_url must start with http:// or https://: {}",
                self.judge.base_url
            )));
        }
        if self.judge.api_key == "sk-xxx" {
            tracing::warn!(
                "judge ({}) api_key is the placeholder 'sk-xxx' — set OPENFUSION_API_KEY env var or real key in config",
                self.judge.model
            );
        }
        Ok(())
    }

    /// Resolve API key for a specific endpoint. Priority:
    /// 1. Endpoint's own api_key (if not placeholder)
    /// 2. OPENFUSION_API_KEY env var
    pub fn resolve_api_key(&self, endpoint: &ModelEndpoint) -> Option<String> {
        if !endpoint.api_key.is_empty() && !endpoint.api_key.starts_with("sk-xxx") {
            return Some(endpoint.api_key.clone());
        }
        std::env::var("OPENFUSION_API_KEY").ok()
    }

    /// Get the effective URL path for a given protocol.
    pub fn protocol_path(protocol: &ApiProtocol, model: &str) -> String {
        match protocol {
            ApiProtocol::OpenaiCompletions => "/v1/chat/completions".into(),
            ApiProtocol::OpenaiResponses => "/v1/responses".into(),
            ApiProtocol::GoogleGenerativeAi => format!("/v1beta/models/{model}:generateContent"),
            ApiProtocol::AnthropicMessages => "/v1/messages".into(),
        }
    }
}
