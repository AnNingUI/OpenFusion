use thiserror::Error;

pub const DEFAULT_TRANSIENT_RETRIES: u32 = 2;

#[derive(Error, Debug)]
pub enum OpenFusionError {
    #[error(
        "No API key configured. Set api_key in config, OPENFUSION_API_KEY env var, or pass via request."
    )]
    NoApiKey,

    #[error("API error: HTTP {status}: {body}")]
    ApiError {
        status: u16,
        body: String,
        retry_after_secs: Option<u64>,
    },

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Worker '{model}' timed out after {duration_ms}ms")]
    WorkerTimeout { model: String, duration_ms: u64 },

    #[error("Worker '{model}' failed: {message}")]
    WorkerError { model: String, message: String },

    #[error("Insufficient workers: {success_count}/{total} succeeded, need {min}\n{details}")]
    InsufficientWorkers {
        success_count: usize,
        total: usize,
        min: usize,
        details: String,
    },

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

impl OpenFusionError {
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            OpenFusionError::ApiError {
                retry_after_secs, ..
            } => *retry_after_secs,
            _ => None,
        }
    }
}

pub fn is_transient_upstream_error(status: u16, body: &str) -> bool {
    if matches!(status, 408 | 425 | 429 | 500 | 502 | 503 | 504 | 529) {
        return true;
    }

    let body = body.to_ascii_lowercase();
    [
        "high demand",
        "temporary errors",
        "temporarily unavailable",
        "try again",
        "too many requests",
        "rate limit",
        "rate_limit",
        "overloaded",
        "overload",
        "capacity",
        "service unavailable",
    ]
    .iter()
    .any(|needle| body.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_demand_message_is_transient_even_without_5xx_status() {
        assert!(is_transient_upstream_error(
            429,
            "We're currently experiencing high demand, which may cause temporary errors."
        ));
        assert!(is_transient_upstream_error(
            400,
            "We're currently experiencing high demand, which may cause temporary errors."
        ));
        assert!(!is_transient_upstream_error(
            400,
            "Invalid request: missing model"
        ));
    }
}
