use thiserror::Error;

#[derive(Error, Debug)]
pub enum OpenFusionError {
    #[error("No API key configured. Set api_key in config, OPENFUSION_API_KEY env var, or pass via request.")]
    NoApiKey,

    #[error("API error: HTTP {status}: {body}")]
    ApiError { status: u16, body: String },

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Worker '{model}' timed out after {duration_ms}ms")]
    WorkerTimeout { model: String, duration_ms: u64 },

    #[error("Worker '{model}' failed: {message}")]
    WorkerError { model: String, message: String },

    #[error("Insufficient workers: {success_count}/{total} succeeded, need {min}")]
    InsufficientWorkers { success_count: usize, total: usize, min: usize },

    #[error("Config error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Session not found: {id}")]
    SessionNotFound { id: String },

    #[error("Invalid protocol conversion: {0}")]
    Protocol(String),
}
