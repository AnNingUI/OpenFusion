use crate::error::OpenFusionError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub fusion: FusionConfig,
    pub judge: ModelEndpoint,
    #[serde(default)]
    pub workers: Vec<ModelEndpoint>,
    #[serde(default)]
    pub storage: StorageConfig,
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
    /// Optional translation model reserved for future advisory prompt shaping.
    /// If None, reuses the judge model.
    #[serde(default)]
    pub translation: Option<ModelEndpoint>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelEndpoint {
    pub base_url: String,
    pub api: ApiProtocol,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub personality: Vec<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
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

fn default_fusion_name() -> String {
    "openfusion/fusion".into()
}

fn default_fusion_timeout() -> u64 {
    120
}

fn default_worker_timeout() -> u64 {
    60
}

fn default_min_workers() -> usize {
    1
}

fn default_sessions_dir() -> PathBuf {
    dirs_home().join(".openfusion").join("sessions")
}

fn default_max_sessions() -> usize {
    1000
}

fn dirs_home() -> PathBuf {
    directories::UserDirs::new()
        .map(|d| d.home_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            name: default_fusion_name(),
            timeout_secs: default_fusion_timeout(),
            worker_timeout_secs: default_worker_timeout(),
            min_workers: default_min_workers(),
            retry: 0,
            translation: None,
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
            let toml_str = toml::to_string_pretty(&default_config).map_err(|e| {
                OpenFusionError::Config(format!("Failed to serialize default config: {e}"))
            })?;
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
            fusion: FusionConfig::default(),
            judge: ModelEndpoint {
                base_url: "https://api.openai.com".into(),
                api: ApiProtocol::OpenaiResponses,
                api_key: "sk-xxx".into(),
                model: "gpt-5.4-mini".into(),
                name: None,
                personality: vec![],
                system_prompt: None,
            },
            workers: vec![
                ModelEndpoint {
                    base_url: "https://api.openai.com".into(),
                    api: ApiProtocol::OpenaiResponses,
                    api_key: "sk-xxx".into(),
                    model: "gpt-5.4-mini".into(),
                    name: Some("analytical".into()),
                    personality: vec!["rational".into(), "logical".into()],
                    system_prompt: None,
                },
                ModelEndpoint {
                    base_url: "https://api.anthropic.com".into(),
                    api: ApiProtocol::AnthropicMessages,
                    api_key: "sk-xxx".into(),
                    model: "claude-sonnet-4.6".into(),
                    name: Some("critical".into()),
                    personality: vec!["strict".into(), "critical".into()],
                    system_prompt: None,
                },
            ],
            storage: StorageConfig::default(),
        }
    }

    fn validate(&self) -> Result<(), OpenFusionError> {
        if self.workers.is_empty() {
            return Err(OpenFusionError::Config(
                "At least one worker is required".into(),
            ));
        }
        if self.fusion.min_workers > self.workers.len() {
            return Err(OpenFusionError::Config(format!(
                "min_workers ({}) exceeds total workers ({})",
                self.fusion.min_workers,
                self.workers.len()
            )));
        }
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

        validate_endpoint("judge", &self.judge)?;
        for (idx, worker) in self.workers.iter().enumerate() {
            validate_endpoint(&format!("worker[{idx}]"), worker)?;
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

    pub fn protocol_path(protocol: &ApiProtocol, model: &str) -> String {
        match protocol {
            ApiProtocol::OpenaiCompletions => "/v1/chat/completions".into(),
            ApiProtocol::OpenaiResponses => "/v1/responses".into(),
            ApiProtocol::GoogleGenerativeAi => format!("/v1beta/models/{model}:generateContent"),
            ApiProtocol::AnthropicMessages => "/v1/messages".into(),
        }
    }
}

fn validate_endpoint(label: &str, endpoint: &ModelEndpoint) -> Result<(), OpenFusionError> {
    if !endpoint.base_url.starts_with("http://") && !endpoint.base_url.starts_with("https://") {
        return Err(OpenFusionError::Config(format!(
            "{label} base_url must start with http:// or https://: {}",
            endpoint.base_url
        )));
    }
    if endpoint.api_key == "sk-xxx" {
        tracing::warn!(
            "{} ({}) api_key is the placeholder 'sk-xxx' - set OPENFUSION_API_KEY env var or real key in config",
            label,
            endpoint.model
        );
    }
    Ok(())
}
