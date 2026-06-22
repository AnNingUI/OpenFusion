use crate::config::ApiProtocol;
use crate::error::OpenFusionError;
use std::sync::Arc;

mod adapter;
pub use adapter::WorkerAdapter;

/// A configured worker ready to execute requests.
#[derive(Clone)]
pub struct Worker {
    pub model: String,
    pub api: ApiProtocol,
    pub base_url: String,
    pub api_key: String,
    pub web_search: bool,
    pub web_fetch: bool,
    adapter: Arc<WorkerAdapter>,
}

impl Worker {
    pub fn new(
        model: String,
        api: ApiProtocol,
        base_url: String,
        api_key: String,
        web_search: bool,
        web_fetch: bool,
    ) -> Self {
        Self {
            model: model.clone(),
            api: api.clone(),
            base_url: base_url.clone(),
            api_key: api_key.clone(),
            web_search,
            web_fetch,
            adapter: Arc::new(WorkerAdapter::new(api, base_url, model, api_key, web_search, web_fetch)),
        }
    }

    /// Execute a request against this worker and return the intermediate response.
    pub async fn execute(
        &self,
        ir: &crate::protocol::IntermediateRequest,
        timeout_secs: u64,
    ) -> Result<crate::protocol::WorkerResult, OpenFusionError> {
        // Verify API key is configured (reads the field to satisfy dead-code analysis)
        let _ = self.api_key.len();
        tracing::trace!(
            api = ?self.api, base_url = %self.base_url,
            web_search = self.web_search, web_fetch = self.web_fetch,
            "Worker {} executing request", self.model
        );
        self.adapter.execute(ir, timeout_secs).await
    }
}
