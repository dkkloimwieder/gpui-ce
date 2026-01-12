//! Web/WASM platform implementation for GPUI
//!
//! This module provides browser-based platform support using WebGPU for rendering
//! and web APIs for windowing, events, and text.

mod dispatcher;
mod platform;
mod window;

pub(crate) use platform::WebPlatform;
pub(crate) use platform::current_platform;
pub(crate) use window::WebWindow;

/// Screen capture is not supported on WASM
pub(crate) type PlatformScreenCaptureFrame = ();
