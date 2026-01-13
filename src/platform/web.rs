//! Web/WASM platform implementation for GPUI
//!
//! This module provides browser-based platform support using WebGPU for rendering
//! and web APIs for windowing, events, and text.

mod dispatcher;
pub(crate) mod events;
pub mod event_listeners;
mod platform;
mod renderer;
mod text_system;
mod web_atlas;
mod window;

pub(crate) use platform::WebPlatform;
pub(crate) use platform::current_platform;
pub use platform::DEFAULT_CANVAS_ID;
#[cfg(target_arch = "wasm32")]
pub use platform::get_canvas_element;
pub use renderer::{GlobalParams, WebRenderer, WebRendererState, WebSurfaceConfig};
pub use web_atlas::{WebGpuAtlas, WebAtlasTextureInfo};
pub(crate) use window::WebWindow;

/// Screen capture is not supported on WASM
pub(crate) type PlatformScreenCaptureFrame = ();
