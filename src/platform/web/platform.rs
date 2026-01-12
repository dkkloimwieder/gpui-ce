//! Web platform implementation - stubs for WASM compilation
//!
//! This provides minimal implementations to allow gpui-ce to compile
//! to wasm32-unknown-unknown. Real implementations will be added in follow-up work.

use crate::{
    AnyWindowHandle, BackgroundExecutor, ClipboardItem, CursorStyle, ForegroundExecutor, Keymap,
    NoopTextSystem, Platform, PlatformDisplay, PlatformKeyboardLayout, PlatformKeyboardMapper,
    PlatformTextSystem, PlatformWindow, Task, WindowAppearance, WindowParams,
    DummyKeyboardMapper, Bounds, Pixels, DisplayId, point, px,
};
use anyhow::Result;
use futures::channel::oneshot;
use parking_lot::Mutex;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};
use uuid::Uuid;

/// Web platform implementation for WASM
pub(crate) struct WebPlatform {
    background_executor: BackgroundExecutor,
    foreground_executor: ForegroundExecutor,
    text_system: Arc<dyn PlatformTextSystem>,
    clipboard: Mutex<Option<ClipboardItem>>,
}

impl WebPlatform {
    pub fn new(background_executor: BackgroundExecutor, foreground_executor: ForegroundExecutor) -> Rc<Self> {
        Rc::new(Self {
            background_executor,
            foreground_executor,
            text_system: Arc::new(NoopTextSystem::new()),
            clipboard: Mutex::new(None),
        })
    }
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
        on_finish_launching();
        // TODO: Set up requestAnimationFrame loop
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
        vec![Rc::new(WebDisplay)]
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(Rc::new(WebDisplay))
    }

    fn active_window(&self) -> Option<AnyWindowHandle> {
        None // TODO
    }

    fn open_window(
        &self,
        _handle: AnyWindowHandle,
        _options: WindowParams,
    ) -> Result<Box<dyn PlatformWindow>> {
        anyhow::bail!("WebPlatform::open_window not yet implemented")
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
struct WebDisplay;

impl PlatformDisplay for WebDisplay {
    fn id(&self) -> DisplayId {
        DisplayId(0)
    }

    fn uuid(&self) -> Result<Uuid> {
        Ok(Uuid::nil())
    }

    fn bounds(&self) -> Bounds<Pixels> {
        // TODO: Get actual window.innerWidth/innerHeight
        Bounds {
            origin: point(px(0.0), px(0.0)),
            size: crate::size(px(1920.0), px(1080.0)),
        }
    }
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
    // TODO: Create proper executors for WASM
    unimplemented!("WebPlatform requires async initialization - use current_platform_async()")
}
