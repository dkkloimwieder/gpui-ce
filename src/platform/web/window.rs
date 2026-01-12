//! Web window implementation for GPUI
//!
//! Implements PlatformWindow using HTML canvas and browser APIs.

use crate::{
    AtlasKey, AtlasTextureId, AtlasTile, Bounds, DevicePixels, DispatchEventResult, GpuSpecs,
    Modifiers, Capslock, Pixels, PlatformAtlas, PlatformDisplay, PlatformInput, PlatformInputHandler,
    PlatformWindow, Point, PromptButton, PromptLevel, RequestFrameOptions, Scene, Size, TileId,
    WindowAppearance, WindowBackgroundAppearance, WindowBounds, WindowControlArea, WindowParams,
    point, px, size,
};
use collections::HashMap;
use futures::channel::oneshot;
use parking_lot::Mutex;
use raw_window_handle::{
    HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    WebDisplayHandle, WebWindowHandle,
};
use std::{
    rc::Rc,
    sync::Arc,
};

/// Web window state
pub(crate) struct WebWindowState {
    /// Window handle for GPUI
    pub(crate) handle: crate::AnyWindowHandle,
    /// Window bounds in pixels
    pub(crate) bounds: Bounds<Pixels>,
    /// Current scale factor (device pixel ratio)
    pub(crate) scale_factor: f32,
    /// Canvas element ID
    pub(crate) canvas_id: u32,
    /// Callbacks
    pub(crate) request_frame_callback: Option<Box<dyn FnMut(RequestFrameOptions)>>,
    pub(crate) input_callback: Option<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    pub(crate) active_status_change_callback: Option<Box<dyn FnMut(bool)>>,
    pub(crate) hover_status_change_callback: Option<Box<dyn FnMut(bool)>>,
    pub(crate) resize_callback: Option<Box<dyn FnMut(Size<Pixels>, f32)>>,
    pub(crate) moved_callback: Option<Box<dyn FnMut()>>,
    pub(crate) should_close_callback: Option<Box<dyn FnMut() -> bool>>,
    pub(crate) close_callback: Option<Box<dyn FnOnce()>>,
    pub(crate) appearance_change_callback: Option<Box<dyn FnMut()>>,
    pub(crate) hit_test_callback: Option<Box<dyn FnMut() -> Option<WindowControlArea>>>,
    pub(crate) input_handler: Option<PlatformInputHandler>,
    /// Display reference
    pub(crate) display: Rc<dyn PlatformDisplay>,
    /// Sprite atlas for this window
    pub(crate) sprite_atlas: Arc<dyn PlatformAtlas>,
    /// Window title
    pub(crate) title: String,
    /// Whether window is active
    pub(crate) is_active: bool,
    /// Whether window is hovered
    pub(crate) is_hovered: bool,
    /// Whether window is fullscreen
    pub(crate) is_fullscreen: bool,
    /// Current mouse position
    pub(crate) mouse_position: Point<Pixels>,
    /// Current modifiers
    pub(crate) modifiers: Modifiers,
}

/// Web window - wraps browser canvas element
#[derive(Clone)]
pub(crate) struct WebWindow(pub(crate) Rc<Mutex<WebWindowState>>);

impl WebWindow {
    /// Create a new web window backed by a canvas element
    pub fn new(
        handle: crate::AnyWindowHandle,
        params: WindowParams,
        display: Rc<dyn PlatformDisplay>,
        canvas_id: u32,
    ) -> Self {
        // Get initial size from params or use display bounds
        let bounds = params.bounds;

        // Get device pixel ratio for scale factor
        let scale_factor = get_device_pixel_ratio();

        Self(Rc::new(Mutex::new(WebWindowState {
            handle,
            bounds,
            scale_factor,
            canvas_id,
            request_frame_callback: None,
            input_callback: None,
            active_status_change_callback: None,
            hover_status_change_callback: None,
            resize_callback: None,
            moved_callback: None,
            should_close_callback: None,
            close_callback: None,
            appearance_change_callback: None,
            hit_test_callback: None,
            input_handler: None,
            display,
            sprite_atlas: Arc::new(WebAtlas::new()),
            title: String::new(),
            is_active: true,
            is_hovered: false,
            is_fullscreen: false,
            mouse_position: point(px(0.0), px(0.0)),
            modifiers: Modifiers::default(),
        })))
    }

    /// Called when browser window is resized
    pub fn handle_resize(&self, width: f32, height: f32) {
        let mut state = self.0.lock();
        let new_size = size(px(width), px(height));
        state.bounds.size = new_size;
        let scale_factor = state.scale_factor;

        if let Some(callback) = state.resize_callback.take() {
            drop(state);
            let mut callback = callback;
            callback(new_size, scale_factor);
            self.0.lock().resize_callback = Some(callback);
        }
    }

    /// Called when mouse moves
    pub fn handle_mouse_move(&self, x: f32, y: f32) {
        let mut state = self.0.lock();
        state.mouse_position = point(px(x), px(y));
    }

    /// Request next animation frame
    pub fn request_frame(&self) {
        let mut state = self.0.lock();
        if let Some(callback) = state.request_frame_callback.take() {
            drop(state);
            let mut callback = callback;
            callback(RequestFrameOptions {
                require_presentation: true,
                force_render: false,
            });
            self.0.lock().request_frame_callback = Some(callback);
        }
    }
}

impl HasWindowHandle for WebWindow {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let state = self.0.lock();
        let mut handle = WebWindowHandle::new(state.canvas_id);
        // SAFETY: The handle is valid for the lifetime of the window
        unsafe {
            Ok(raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::Web(handle)))
        }
    }
}

impl HasDisplayHandle for WebWindow {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let handle = WebDisplayHandle::new();
        // SAFETY: The handle is valid for the lifetime of the window
        unsafe {
            Ok(raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::Web(handle)))
        }
    }
}

impl PlatformWindow for WebWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        self.0.lock().bounds
    }

    fn is_maximized(&self) -> bool {
        false
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Windowed(self.bounds())
    }

    fn content_size(&self) -> Size<Pixels> {
        self.bounds().size
    }

    fn resize(&mut self, size: Size<Pixels>) {
        self.0.lock().bounds.size = size;
    }

    fn scale_factor(&self) -> f32 {
        self.0.lock().scale_factor
    }

    fn appearance(&self) -> WindowAppearance {
        // Check prefers-color-scheme media query
        if prefers_dark_mode() {
            WindowAppearance::Dark
        } else {
            WindowAppearance::Light
        }
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.0.lock().display.clone())
    }

    fn mouse_position(&self) -> Point<Pixels> {
        self.0.lock().mouse_position
    }

    fn modifiers(&self) -> Modifiers {
        self.0.lock().modifiers
    }

    fn capslock(&self) -> Capslock {
        Capslock::default()
    }

    fn set_input_handler(&mut self, input_handler: PlatformInputHandler) {
        self.0.lock().input_handler = Some(input_handler);
    }

    fn take_input_handler(&mut self) -> Option<PlatformInputHandler> {
        self.0.lock().input_handler.take()
    }

    fn prompt(
        &self,
        _level: PromptLevel,
        _msg: &str,
        _detail: Option<&str>,
        _answers: &[PromptButton],
    ) -> Option<oneshot::Receiver<usize>> {
        // Use browser's confirm/alert dialogs
        // For now, just return first answer
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(0);
        Some(rx)
    }

    fn activate(&self) {
        let mut state = self.0.lock();
        state.is_active = true;
        if let Some(callback) = state.active_status_change_callback.take() {
            drop(state);
            let mut callback = callback;
            callback(true);
            self.0.lock().active_status_change_callback = Some(callback);
        }
    }

    fn is_active(&self) -> bool {
        self.0.lock().is_active
    }

    fn is_hovered(&self) -> bool {
        self.0.lock().is_hovered
    }

    fn set_title(&mut self, title: &str) {
        self.0.lock().title = title.to_string();
        // Could update document.title via web-sys
    }

    fn set_background_appearance(&self, _background: WindowBackgroundAppearance) {
        // Could be implemented via canvas styling
    }

    fn minimize(&self) {
        // Not applicable in browser
    }

    fn zoom(&self) {
        // Not applicable in browser
    }

    fn toggle_fullscreen(&self) {
        let mut state = self.0.lock();
        state.is_fullscreen = !state.is_fullscreen;
        // Could use Fullscreen API via web-sys
    }

    fn is_fullscreen(&self) -> bool {
        self.0.lock().is_fullscreen
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        self.0.lock().request_frame_callback = Some(callback);
    }

    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {
        self.0.lock().input_callback = Some(callback);
    }

    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.0.lock().active_status_change_callback = Some(callback);
    }

    fn on_hover_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        self.0.lock().hover_status_change_callback = Some(callback);
    }

    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        self.0.lock().resize_callback = Some(callback);
    }

    fn on_moved(&self, callback: Box<dyn FnMut()>) {
        self.0.lock().moved_callback = Some(callback);
    }

    fn on_should_close(&self, callback: Box<dyn FnMut() -> bool>) {
        self.0.lock().should_close_callback = Some(callback);
    }

    fn on_close(&self, callback: Box<dyn FnOnce()>) {
        self.0.lock().close_callback = Some(callback);
    }

    fn on_hit_test_window_control(&self, callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
        self.0.lock().hit_test_callback = Some(callback);
    }

    fn on_appearance_changed(&self, callback: Box<dyn FnMut()>) {
        self.0.lock().appearance_change_callback = Some(callback);
    }

    fn draw(&self, _scene: &Scene) {
        // Drawing is handled by the blade renderer
        // This callback is used by GPUI to trigger scene rendering
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.0.lock().sprite_atlas.clone()
    }

    fn gpu_specs(&self) -> Option<GpuSpecs> {
        // Could query WebGPU adapter info
        Some(GpuSpecs {
            is_software_emulated: false,
            device_name: "WebGPU".to_string(),
            driver_name: "Browser".to_string(),
            driver_info: "WebGPU".to_string(),
        })
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {
        // Could position an IME overlay element
    }
}

//=============================================================================
// Web Atlas Implementation
//=============================================================================

/// Simple atlas for web - similar to TestAtlas
pub(crate) struct WebAtlasState {
    next_id: u32,
    tiles: HashMap<AtlasKey, AtlasTile>,
}

pub(crate) struct WebAtlas(Mutex<WebAtlasState>);

impl WebAtlas {
    pub fn new() -> Self {
        WebAtlas(Mutex::new(WebAtlasState {
            next_id: 0,
            tiles: HashMap::default(),
        }))
    }
}

impl PlatformAtlas for WebAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>>,
    ) -> anyhow::Result<Option<AtlasTile>> {
        let mut state = self.0.lock();
        if let Some(tile) = state.tiles.get(key) {
            return Ok(Some(tile.clone()));
        }
        drop(state);

        let Some((size, _)) = build()? else {
            return Ok(None);
        };

        let mut state = self.0.lock();
        state.next_id += 1;
        let texture_id = state.next_id;
        state.next_id += 1;
        let tile_id = state.next_id;

        state.tiles.insert(
            key.clone(),
            AtlasTile {
                texture_id: AtlasTextureId {
                    index: texture_id,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: TileId(tile_id),
                padding: 0,
                bounds: Bounds {
                    origin: Point::default(),
                    size,
                },
            },
        );

        Ok(Some(state.tiles[key].clone()))
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().tiles.remove(key);
    }
}

//=============================================================================
// Helper Functions
//=============================================================================

/// Get device pixel ratio from browser
fn get_device_pixel_ratio() -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .map(|w| w.device_pixel_ratio() as f32)
            .unwrap_or(1.0)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        1.0
    }
}

/// Check if user prefers dark mode
fn prefers_dark_mode() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.match_media("(prefers-color-scheme: dark)").ok())
            .flatten()
            .map(|m| m.matches())
            .unwrap_or(false)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}
