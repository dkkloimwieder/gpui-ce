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

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(target_arch = "wasm32")]
use super::WebRenderer;

/// Web window state
pub(crate) struct WebWindowState {
    /// Window handle for GPUI
    pub(crate) handle: crate::AnyWindowHandle,
    /// Window bounds in pixels
    pub(crate) bounds: Bounds<Pixels>,
    /// Current scale factor (device pixel ratio)
    pub(crate) scale_factor: f32,
    /// Canvas element ID (used for raw_window_handle)
    pub(crate) canvas_id: u32,
    /// Canvas element reference (for WASM target only)
    #[cfg(target_arch = "wasm32")]
    pub(crate) canvas: Option<web_sys::HtmlCanvasElement>,
    /// WebGPU renderer (set after async initialization)
    #[cfg(target_arch = "wasm32")]
    pub(crate) renderer: Option<WebRenderer>,
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
    /// Sprite atlas for this window (fallback when renderer not initialized)
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
    /// Last mouse down time for click counting
    pub(crate) last_mouse_down_time: Option<f64>,
    /// Last mouse down button
    pub(crate) last_mouse_down_button: Option<i16>,
    /// Current click count
    pub(crate) click_count: usize,
    /// Event listeners (must be kept alive)
    #[cfg(target_arch = "wasm32")]
    pub(crate) event_listeners: Option<super::event_listeners::EventListeners>,
}

/// Web window - wraps browser canvas element
#[derive(Clone)]
pub(crate) struct WebWindow(pub(crate) Rc<Mutex<WebWindowState>>);

impl WebWindow {
    /// Create a new web window backed by a canvas element
    #[cfg(target_arch = "wasm32")]
    pub fn new(
        handle: crate::AnyWindowHandle,
        params: WindowParams,
        display: Rc<dyn PlatformDisplay>,
        canvas_id: u32,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Self {
        // Get device pixel ratio for scale factor
        let scale_factor = get_device_pixel_ratio();

        // Get canvas size or use params bounds
        let client_width = canvas.client_width() as f32;
        let client_height = canvas.client_height() as f32;

        // If client size is 0, use the canvas buffer size or fallback to reasonable defaults
        let (width, height) = if client_width > 0.0 && client_height > 0.0 {
            (client_width, client_height)
        } else {
            // Canvas not laid out yet, use buffer size or params
            let buf_width = canvas.width() as f32;
            let buf_height = canvas.height() as f32;
            if buf_width > 0.0 && buf_height > 0.0 {
                (buf_width, buf_height)
            } else {
                // Use params bounds size as fallback
                (params.bounds.size.width.0, params.bounds.size.height.0)
            }
        };

        log::info!(
            "WebWindow::new: canvas client={}x{}, buffer={}x{}, using size={}x{}, scale={}",
            client_width, client_height,
            canvas.width(), canvas.height(),
            width, height, scale_factor
        );

        // Set canvas dimensions to match client size with device pixel ratio
        let device_width = (width * scale_factor) as u32;
        let device_height = (height * scale_factor) as u32;
        canvas.set_width(device_width);
        canvas.set_height(device_height);

        log::info!("WebWindow::new: set canvas buffer to {}x{}", device_width, device_height);

        // Use canvas size for bounds
        let bounds = crate::Bounds {
            origin: params.bounds.origin,
            size: size(px(width), px(height)),
        };

        Self(Rc::new(Mutex::new(WebWindowState {
            handle,
            bounds,
            scale_factor,
            canvas_id,
            canvas: Some(canvas),
            renderer: None,
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
            last_mouse_down_time: None,
            last_mouse_down_button: None,
            click_count: 0,
            event_listeners: None,
        })))
    }

    /// Create a new web window (non-WASM fallback for testing)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        handle: crate::AnyWindowHandle,
        params: WindowParams,
        display: Rc<dyn PlatformDisplay>,
        canvas_id: u32,
    ) -> Self {
        let bounds = params.bounds;
        let scale_factor = 1.0;

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
            last_mouse_down_time: None,
            last_mouse_down_button: None,
            click_count: 0,
        })))
    }

    /// Get the canvas element (WASM only)
    #[cfg(target_arch = "wasm32")]
    pub fn canvas(&self) -> Option<web_sys::HtmlCanvasElement> {
        self.0.lock().canvas.clone()
    }

    /// Set up browser event listeners for this window
    #[cfg(target_arch = "wasm32")]
    pub fn setup_event_listeners(self: &Rc<Self>) {
        if let Some(canvas) = self.canvas() {
            match super::event_listeners::setup_event_listeners(&canvas, Rc::new(self.as_ref().clone())) {
                Ok(listeners) => {
                    self.0.lock().event_listeners = Some(listeners);
                    log::info!("Event listeners set up successfully");
                }
                Err(e) => {
                    log::error!("Failed to set up event listeners: {:?}", e);
                }
            }
        } else {
            log::error!("No canvas to set up event listeners on");
        }
    }

    /// Set the WebGPU renderer after async initialization
    ///
    /// This must be called after the renderer is initialized asynchronously.
    /// Until set, draw() will be a no-op.
    #[cfg(target_arch = "wasm32")]
    pub fn set_renderer(&self, renderer: WebRenderer) {
        self.0.lock().renderer = Some(renderer);
    }

    /// Get the renderer if initialized
    #[cfg(target_arch = "wasm32")]
    pub fn renderer(&self) -> Option<WebRenderer> {
        self.0.lock().renderer.clone()
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

    //=========================================================================
    // Input Event Handling
    //=========================================================================

    /// Dispatch a PlatformInput event through the input callback
    pub fn dispatch_input(&self, input: PlatformInput) -> crate::DispatchEventResult {
        let mut state = self.0.lock();
        if let Some(callback) = state.input_callback.take() {
            drop(state);
            log::debug!("Dispatching input event: {:?}", std::mem::discriminant(&input));
            let mut callback = callback;
            let result = callback(input);
            log::debug!("Input event dispatched, result: {:?}", result);
            self.0.lock().input_callback = Some(callback);
            result
        } else {
            log::warn!("No input callback registered - event dropped");
            crate::DispatchEventResult::default()
        }
    }

    /// Handle browser mousedown event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_mouse_down(&self, event: &web_sys::MouseEvent, now: f64) {
        use super::events::{modifiers_from_mouse_event, mouse_button_from_browser};

        let button = event.button();
        let mut state = self.0.lock();

        // Calculate click count (double-click detection)
        // Double-click if same button within 500ms
        const DOUBLE_CLICK_MS: f64 = 500.0;
        if state.last_mouse_down_button == Some(button) {
            if let Some(last_time) = state.last_mouse_down_time {
                if now - last_time < DOUBLE_CLICK_MS {
                    state.click_count += 1;
                } else {
                    state.click_count = 1;
                }
            } else {
                state.click_count = 1;
            }
        } else {
            state.click_count = 1;
        }
        state.last_mouse_down_time = Some(now);
        state.last_mouse_down_button = Some(button);

        let click_count = state.click_count;
        state.mouse_position = point(px(event.offset_x() as f32), px(event.offset_y() as f32));
        state.modifiers = modifiers_from_mouse_event(event);

        drop(state);

        let input = PlatformInput::MouseDown(crate::MouseDownEvent {
            button: mouse_button_from_browser(button),
            position: point(px(event.offset_x() as f32), px(event.offset_y() as f32)),
            modifiers: modifiers_from_mouse_event(event),
            click_count,
            first_mouse: false,
        });

        self.dispatch_input(input);
    }

    /// Handle browser mouseup event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_mouse_up(&self, event: &web_sys::MouseEvent) {
        use super::events::{modifiers_from_mouse_event, mouse_button_from_browser};

        let state = self.0.lock();
        let click_count = state.click_count;
        drop(state);

        let input = PlatformInput::MouseUp(crate::MouseUpEvent {
            button: mouse_button_from_browser(event.button()),
            position: point(px(event.offset_x() as f32), px(event.offset_y() as f32)),
            modifiers: modifiers_from_mouse_event(event),
            click_count,
        });

        self.dispatch_input(input);
    }

    /// Handle browser mousemove event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_mouse_move_event(&self, event: &web_sys::MouseEvent) {
        use super::events::{modifiers_from_mouse_event, pressed_button_from_buttons};

        let position = point(px(event.offset_x() as f32), px(event.offset_y() as f32));

        {
            let mut state = self.0.lock();
            state.mouse_position = position;
            state.modifiers = modifiers_from_mouse_event(event);
        }

        let input = PlatformInput::MouseMove(crate::MouseMoveEvent {
            position,
            pressed_button: pressed_button_from_buttons(event.buttons()),
            modifiers: modifiers_from_mouse_event(event),
        });

        self.dispatch_input(input);
    }

    /// Handle browser wheel event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_wheel(&self, event: &web_sys::WheelEvent) {
        use super::events::scroll_wheel_from_browser;

        let input = PlatformInput::ScrollWheel(scroll_wheel_from_browser(event));
        self.dispatch_input(input);
    }

    /// Handle browser keydown event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_key_down(&self, event: &web_sys::KeyboardEvent) {
        use super::events::{is_modifier_key, key_down_from_browser, modifiers_changed_from_keyboard, modifiers_from_keyboard_event};

        // Update modifiers
        {
            let mut state = self.0.lock();
            state.modifiers = modifiers_from_keyboard_event(event);
        }

        // If this is a modifier key, send ModifiersChanged event
        if is_modifier_key(event) {
            let input = PlatformInput::ModifiersChanged(modifiers_changed_from_keyboard(event));
            self.dispatch_input(input);
        } else {
            let input = PlatformInput::KeyDown(key_down_from_browser(event));
            self.dispatch_input(input);
        }
    }

    /// Handle browser keyup event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_key_up(&self, event: &web_sys::KeyboardEvent) {
        use super::events::{is_modifier_key, key_up_from_browser, modifiers_changed_from_keyboard, modifiers_from_keyboard_event};

        // Update modifiers
        {
            let mut state = self.0.lock();
            state.modifiers = modifiers_from_keyboard_event(event);
        }

        // If this is a modifier key, send ModifiersChanged event
        if is_modifier_key(event) {
            let input = PlatformInput::ModifiersChanged(modifiers_changed_from_keyboard(event));
            self.dispatch_input(input);
        } else {
            let input = PlatformInput::KeyUp(key_up_from_browser(event));
            self.dispatch_input(input);
        }
    }

    /// Handle browser mouseenter event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_mouse_enter(&self) {
        let mut state = self.0.lock();
        state.is_hovered = true;
        if let Some(callback) = state.hover_status_change_callback.take() {
            drop(state);
            let mut callback = callback;
            callback(true);
            self.0.lock().hover_status_change_callback = Some(callback);
        }
    }

    /// Handle browser mouseleave event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_mouse_leave(&self, event: &web_sys::MouseEvent) {
        use super::events::{modifiers_from_mouse_event, pressed_button_from_buttons};

        {
            let mut state = self.0.lock();
            state.is_hovered = false;
        }

        // Send MouseExited event
        let input = PlatformInput::MouseExited(crate::MouseExitEvent {
            position: point(px(event.offset_x() as f32), px(event.offset_y() as f32)),
            pressed_button: pressed_button_from_buttons(event.buttons()),
            modifiers: modifiers_from_mouse_event(event),
        });
        self.dispatch_input(input);

        // Call hover status callback
        let mut state = self.0.lock();
        if let Some(callback) = state.hover_status_change_callback.take() {
            drop(state);
            let mut callback = callback;
            callback(false);
            self.0.lock().hover_status_change_callback = Some(callback);
        }
    }

    /// Handle browser focus event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_focus(&self) {
        let mut state = self.0.lock();
        state.is_active = true;
        if let Some(callback) = state.active_status_change_callback.take() {
            drop(state);
            let mut callback = callback;
            callback(true);
            self.0.lock().active_status_change_callback = Some(callback);
        }
    }

    /// Handle browser blur event
    #[cfg(target_arch = "wasm32")]
    pub fn handle_blur(&self) {
        let mut state = self.0.lock();
        state.is_active = false;
        if let Some(callback) = state.active_status_change_callback.take() {
            drop(state);
            let mut callback = callback;
            callback(false);
            self.0.lock().active_status_change_callback = Some(callback);
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

    fn draw(&self, scene: &Scene) {
        #[cfg(target_arch = "wasm32")]
        {
            // Log scene contents for debugging
            let batch_count = scene.batches().count();
            log::debug!("WebWindow::draw called with scene containing {} batches", batch_count);

            // Clone the renderer while holding the lock, then release lock before drawing
            let renderer = {
                let state = self.0.lock();
                state.renderer.clone()
            };
            if let Some(renderer) = renderer {
                renderer.draw(scene);
            } else {
                log::warn!("WebWindow::draw called before renderer is initialized");
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = scene;
            // No-op on non-WASM
        }
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        let state = self.0.lock();
        // Use GPU atlas from renderer if available, otherwise fallback to simple atlas
        #[cfg(target_arch = "wasm32")]
        if let Some(ref renderer) = state.renderer {
            if let Some(atlas) = renderer.sprite_atlas() {
                return atlas;
            }
        }
        state.sprite_atlas.clone()
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
