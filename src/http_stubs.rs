//! HTTP client stubs for WASM
//!
//! These provide the same API surface as http_client but are non-functional stubs.
//! On WASM, actual HTTP will be done via browser fetch API, not these types.

use futures::future::BoxFuture;

/// URL type for HTTP requests
pub type Url = String;

/// HTTP header value
#[derive(Debug, Clone)]
pub struct HeaderValue(String);

impl HeaderValue {
    /// Create a header value from a string
    pub fn from_str(s: &str) -> Result<Self, ()> {
        Ok(HeaderValue(s.to_string()))
    }
}

/// HTTP module re-exports
pub mod http {
    pub use super::HeaderValue;
}

/// Request body
pub struct AsyncBody;

/// HTTP Request
pub struct Request<T> {
    _body: std::marker::PhantomData<T>,
}

/// HTTP Response
pub struct Response<T> {
    _body: std::marker::PhantomData<T>,
}

/// The HttpClient trait for making HTTP requests
///
/// On WASM, this is a stub - actual HTTP requests should use the browser's fetch API
/// through wasm-bindgen/web-sys.
pub trait HttpClient: Send + Sync {
    /// Get client type name
    fn type_name(&self) -> &'static str;

    /// Execute a request
    fn send(
        &self,
        req: Request<AsyncBody>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>>;

    /// Get the user agent
    fn user_agent(&self) -> Option<&http::HeaderValue> {
        None
    }

    /// Get the proxy URL
    fn proxy(&self) -> Option<&Url> {
        None
    }
}
