//! Web/WASM platform implementation for GPUI
//!
//! This module provides browser-based platform support using WebGPU for rendering
//! and web APIs for windowing, events, and text.

mod platform;

pub(crate) use platform::WebPlatform;
pub(crate) use platform::current_platform;
