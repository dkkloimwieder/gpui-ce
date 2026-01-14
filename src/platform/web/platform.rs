//! Web platform implementation for GPUI
//!
//! Provides browser-based platform support using WebGPU for rendering
//! and web APIs for windowing, events, and text.

use super::dispatcher::WebDispatcher;
use super::text_system::WebTextSystem;
use super::window::WebWindow;
use crate::{
    AnyWindowHandle, BackgroundExecutor, ClipboardItem, CursorStyle, ForegroundExecutor, Keymap,
    Platform, PlatformDisplay, PlatformKeyboardLayout, PlatformKeyboardMapper,
    PlatformTextSystem, PlatformWindow, Task, WindowAppearance, WindowParams,
    DummyKeyboardMapper, Bounds, Pixels, DisplayId, point, px,
};
use anyhow::Result;
use futures::channel::oneshot;
use parking_lot::Mutex;
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};
use uuid::Uuid;

// Thread-local storage for the platform instance (WASM is single-threaded)
thread_local! {
    static PLATFORM: RefCell<Option<Rc<WebPlatform>>> = const { RefCell::new(None) };
}

/// Default canvas element ID for GPUI
pub const DEFAULT_CANVAS_ID: &str = "gpui-canvas";

/// Get canvas element from the DOM by ID
#[cfg(target_arch = "wasm32")]
pub fn get_canvas_element(canvas_id: &str) -> Result<web_sys::HtmlCanvasElement> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window()
        .ok_or_else(|| anyhow::anyhow!("No window object"))?;
    let document = window.document()
        .ok_or_else(|| anyhow::anyhow!("No document object"))?;
    let element = document.get_element_by_id(canvas_id)
        .ok_or_else(|| anyhow::anyhow!("Canvas element '{}' not found", canvas_id))?;
    let canvas = element.dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| anyhow::anyhow!("Element '{}' is not a canvas", canvas_id))?;

    Ok(canvas)
}

/// Web platform implementation for WASM
pub(crate) struct WebPlatform {
    background_executor: BackgroundExecutor,
    foreground_executor: ForegroundExecutor,
    text_system: Arc<dyn PlatformTextSystem>,
    clipboard: Mutex<Option<ClipboardItem>>,
    /// Dispatcher for task scheduling
    dispatcher: Arc<WebDispatcher>,
    /// Active window (single window for now)
    active_window: RefCell<Option<WebWindow>>,
    /// Primary display
    display: Rc<WebDisplay>,
    /// Next canvas ID
    next_canvas_id: RefCell<u32>,
    /// Current cursor style
    cursor_style: RefCell<CursorStyle>,
    /// Pre-initialized renderer (set before opening windows)
    #[cfg(target_arch = "wasm32")]
    pending_renderer: RefCell<Option<super::renderer::WebRenderer>>,
}

impl WebPlatform {
    /// Create a new web platform with the given executors
    pub fn new(background_executor: BackgroundExecutor, foreground_executor: ForegroundExecutor) -> Rc<Self> {
        let dispatcher = Arc::new(WebDispatcher::new());
        Rc::new(Self {
            background_executor,
            foreground_executor,
            text_system: Arc::new(WebTextSystem::new()),
            clipboard: Mutex::new(None),
            dispatcher,
            active_window: RefCell::new(None),
            display: Rc::new(WebDisplay::new()),
            next_canvas_id: RefCell::new(1),
            cursor_style: RefCell::new(CursorStyle::Arrow),
            #[cfg(target_arch = "wasm32")]
            pending_renderer: RefCell::new(None),
        })
    }

    /// Get the active web window (if any)
    #[cfg(target_arch = "wasm32")]
    pub fn get_active_web_window(&self) -> Option<WebWindow> {
        self.active_window.borrow().clone()
    }

    /// Set a pre-initialized renderer to be used by windows
    #[cfg(target_arch = "wasm32")]
    pub fn set_pending_renderer(&self, renderer: super::renderer::WebRenderer) {
        *self.pending_renderer.borrow_mut() = Some(renderer);
    }

    /// Take the pending renderer (removes it from storage)
    #[cfg(target_arch = "wasm32")]
    pub fn take_pending_renderer(&self) -> Option<super::renderer::WebRenderer> {
        self.pending_renderer.borrow_mut().take()
    }
}

/// Set a pre-initialized renderer BEFORE opening windows
/// This ensures the GPU atlas is available for text rasterization from the start
#[cfg(target_arch = "wasm32")]
pub fn set_pending_renderer(renderer: super::renderer::WebRenderer) {
    PLATFORM.with(|platform| {
        if let Some(ref p) = *platform.borrow() {
            p.set_pending_renderer(renderer);
            log::info!("Pending renderer set on platform");
        } else {
            log::warn!("Platform not initialized - cannot set pending renderer");
        }
    });
}

/// Set the active renderer on the currently active window
/// This should be called after initializing the WebRenderer
#[cfg(target_arch = "wasm32")]
pub fn set_window_renderer(renderer: super::renderer::WebRenderer) {
    PLATFORM.with(|platform| {
        if let Some(ref p) = *platform.borrow() {
            if let Some(window) = p.get_active_web_window() {
                window.set_renderer(renderer);
                log::info!("Renderer attached to active window");
            } else {
                log::warn!("No active window to attach renderer to");
            }
        } else {
            log::warn!("Platform not initialized");
        }
    });
}

impl Platform for WebPlatform {
    fn background_executor(&self) -> BackgroundExecutor {
        self.background_executor.clone()
    }

    fn foreground_executor(&self) -> ForegroundExecutor {
        self.foreground_executor.clone()
    }

    fn text_system(&self) -> Arc<dyn PlatformTextSystem> {
        self.text_system.clone()
    }

    fn run(&self, on_finish_launching: Box<dyn FnOnce()>) {
        // Run the launch callback
        on_finish_launching();

        // In WASM, we don't block here. The event loop is driven by browser events
        // and requestAnimationFrame. The caller should set up event handlers
        // after calling run().
        //
        // The dispatcher will be polled via requestAnimationFrame callbacks
        // that are set up when windows request frames.
    }

    fn quit(&self) {
        // No-op in browser
    }

    fn restart(&self, _binary_path: Option<PathBuf>) {
        // Could reload the page
    }

    fn activate(&self, _ignoring_other_apps: bool) {}

    fn hide(&self) {}

    fn hide_other_apps(&self) {}

    fn unhide_other_apps(&self) {}

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        vec![self.display.clone()]
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.display.clone())
    }

    fn active_window(&self) -> Option<AnyWindowHandle> {
        self.active_window.borrow().as_ref().map(|w| w.0.lock().handle)
    }

    #[cfg(target_arch = "wasm32")]
    fn open_window(
        &self,
        handle: AnyWindowHandle,
        options: WindowParams,
    ) -> Result<Box<dyn PlatformWindow>> {
        // Generate a unique canvas ID for raw_window_handle
        let canvas_id = {
            let mut id = self.next_canvas_id.borrow_mut();
            let current = *id;
            *id += 1;
            current
        };

        // Get or create canvas element from the DOM
        // For now, use the default canvas ID; future: support multiple canvases
        let canvas = get_canvas_element(DEFAULT_CANVAS_ID)?;

        // Create the web window with the canvas
        let window = WebWindow::new(
            handle,
            options,
            self.display.clone(),
            canvas_id,
            canvas.clone(),
        );

        // If a renderer was pre-initialized, attach it to the window immediately
        // This ensures the GPU atlas is available for text rasterization
        if let Some(renderer) = self.take_pending_renderer() {
            log::info!("Attaching pre-initialized renderer to new window");
            window.set_renderer(renderer);
        }

        // Set up event listeners
        let window_rc = std::rc::Rc::new(window.clone());
        match super::event_listeners::setup_event_listeners(&canvas, window_rc.clone()) {
            Ok(listeners) => {
                window.0.lock().event_listeners = Some(listeners);
                log::info!("Event listeners set up successfully");
            }
            Err(e) => {
                log::error!("Failed to set up event listeners: {:?}", e);
            }
        }

        // Start animation loop for continuous rendering
        match super::event_listeners::start_animation_loop(window_rc) {
            Ok(()) => {
                log::info!("Animation loop started");
            }
            Err(e) => {
                log::error!("Failed to start animation loop: {:?}", e);
            }
        }

        // Store as active window
        *self.active_window.borrow_mut() = Some(window.clone());

        Ok(Box::new(window))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn open_window(
        &self,
        handle: AnyWindowHandle,
        options: WindowParams,
    ) -> Result<Box<dyn PlatformWindow>> {
        // Non-WASM fallback for testing
        let canvas_id = {
            let mut id = self.next_canvas_id.borrow_mut();
            let current = *id;
            *id += 1;
            current
        };

        let window = WebWindow::new(
            handle,
            options,
            self.display.clone(),
            canvas_id,
        );

        *self.active_window.borrow_mut() = Some(window.clone());
        Ok(Box::new(window))
    }

    fn window_appearance(&self) -> WindowAppearance {
        // TODO: Check prefers-color-scheme media query
        WindowAppearance::Light
    }

    fn open_url(&self, _url: &str) {
        // TODO: window.open()
    }

    fn on_open_urls(&self, _callback: Box<dyn FnMut(Vec<String>)>) {}

    fn register_url_scheme(&self, _url: &str) -> Task<Result<()>> {
        Task::ready(Err(anyhow::anyhow!("URL schemes not supported in browser")))
    }

    fn prompt_for_paths(
        &self,
        _options: crate::PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Err(anyhow::anyhow!("File picker not yet implemented")));
        rx
    }

    fn prompt_for_new_path(
        &self,
        _directory: &Path,
        _suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Err(anyhow::anyhow!("File picker not yet implemented")));
        rx
    }

    fn can_select_mixed_files_and_dirs(&self) -> bool {
        false
    }

    fn reveal_path(&self, _path: &Path) {}

    fn open_with_system(&self, _path: &Path) {}

    fn on_quit(&self, _callback: Box<dyn FnMut()>) {}

    fn on_reopen(&self, _callback: Box<dyn FnMut()>) {}

    fn set_menus(&self, _menus: Vec<crate::Menu>, _keymap: &Keymap) {}

    fn set_dock_menu(&self, _menu: Vec<crate::MenuItem>, _keymap: &Keymap) {}

    fn on_app_menu_action(&self, _callback: Box<dyn FnMut(&dyn crate::Action)>) {}

    fn on_will_open_app_menu(&self, _callback: Box<dyn FnMut()>) {}

    fn on_validate_app_menu_command(&self, _callback: Box<dyn FnMut(&dyn crate::Action) -> bool>) {}

    fn app_path(&self) -> Result<PathBuf> {
        Ok(PathBuf::from("/"))
    }

    fn path_for_auxiliary_executable(&self, _name: &str) -> Result<PathBuf> {
        Err(anyhow::anyhow!("No auxiliary executables in browser"))
    }

    fn set_cursor_style(&self, _style: CursorStyle) {
        // TODO: Set CSS cursor
    }

    fn should_auto_hide_scrollbars(&self) -> bool {
        false
    }

    fn write_to_clipboard(&self, item: ClipboardItem) {
        *self.clipboard.lock() = Some(item);
        // TODO: Use navigator.clipboard API
    }

    fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        self.clipboard.lock().clone()
        // TODO: Use navigator.clipboard API
    }

    fn write_credentials(&self, _url: &str, _username: &str, _password: &[u8]) -> Task<Result<()>> {
        Task::ready(Err(anyhow::anyhow!("Credentials not supported in browser")))
    }

    fn read_credentials(&self, _url: &str) -> Task<Result<Option<(String, Vec<u8>)>>> {
        Task::ready(Ok(None))
    }

    fn delete_credentials(&self, _url: &str) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }

    fn keyboard_layout(&self) -> Box<dyn PlatformKeyboardLayout> {
        Box::new(WebKeyboardLayout)
    }

    fn keyboard_mapper(&self) -> Rc<dyn PlatformKeyboardMapper> {
        Rc::new(DummyKeyboardMapper)
    }

    fn on_keyboard_layout_change(&self, _callback: Box<dyn FnMut()>) {}
}

/// Web display - represents the browser viewport
#[derive(Debug)]
pub(crate) struct WebDisplay {
    /// Unique display ID
    id: DisplayId,
}

impl WebDisplay {
    /// Create a new web display
    pub fn new() -> Self {
        Self {
            id: DisplayId(0),
        }
    }
}

impl PlatformDisplay for WebDisplay {
    fn id(&self) -> DisplayId {
        self.id
    }

    fn uuid(&self) -> Result<Uuid> {
        Ok(Uuid::nil())
    }

    fn bounds(&self) -> Bounds<Pixels> {
        #[cfg(target_arch = "wasm32")]
        {
            let (width, height) = get_window_inner_size();
            Bounds {
                origin: point(px(0.0), px(0.0)),
                size: crate::size(px(width), px(height)),
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Bounds {
                origin: point(px(0.0), px(0.0)),
                size: crate::size(px(1920.0), px(1080.0)),
            }
        }
    }
}

/// Get browser window inner size
#[cfg(target_arch = "wasm32")]
fn get_window_inner_size() -> (f32, f32) {
    web_sys::window()
        .and_then(|w| {
            let width = w.inner_width().ok()?.as_f64()? as f32;
            let height = w.inner_height().ok()?.as_f64()? as f32;
            Some((width, height))
        })
        .unwrap_or((1920.0, 1080.0))
}

struct WebKeyboardLayout;

impl PlatformKeyboardLayout for WebKeyboardLayout {
    fn id(&self) -> &str {
        "web.keyboard.default"
    }

    fn name(&self) -> &str {
        "Web Keyboard"
    }
}

pub(crate) fn current_platform(_headless: bool) -> Rc<dyn Platform> {
    PLATFORM.with(|platform| {
        let mut platform_ref = platform.borrow_mut();
        if let Some(ref existing) = *platform_ref {
            return existing.clone() as Rc<dyn Platform>;
        }

        // Create the dispatcher
        let dispatcher = Arc::new(WebDispatcher::new());

        // Create executors from the dispatcher
        let background_executor = BackgroundExecutor::new(dispatcher.clone());
        let foreground_executor = ForegroundExecutor::new(dispatcher);

        // Create and cache the platform
        let web_platform = WebPlatform::new(background_executor, foreground_executor);
        *platform_ref = Some(web_platform.clone());

        web_platform as Rc<dyn Platform>
    })
}
