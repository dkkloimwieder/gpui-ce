//! Browser event listener setup for GPUI
//!
//! This module attaches DOM event listeners to a canvas element and connects
//! them to WebWindow's event handling methods.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
use super::window::WebWindow;

/// Stored closures for event listeners
/// These need to be kept alive for the lifetime of the window
#[cfg(target_arch = "wasm32")]
pub struct EventListeners {
    _mousedown: Closure<dyn FnMut(web_sys::MouseEvent)>,
    _mouseup: Closure<dyn FnMut(web_sys::MouseEvent)>,
    _mousemove: Closure<dyn FnMut(web_sys::MouseEvent)>,
    _mouseenter: Closure<dyn FnMut(web_sys::MouseEvent)>,
    _mouseleave: Closure<dyn FnMut(web_sys::MouseEvent)>,
    _wheel: Closure<dyn FnMut(web_sys::WheelEvent)>,
    _keydown: Closure<dyn FnMut(web_sys::KeyboardEvent)>,
    _keyup: Closure<dyn FnMut(web_sys::KeyboardEvent)>,
    _focus: Closure<dyn FnMut(web_sys::FocusEvent)>,
    _blur: Closure<dyn FnMut(web_sys::FocusEvent)>,
    _resize: Closure<dyn FnMut(web_sys::Event)>,
}

/// Set up all event listeners on a canvas element
///
/// Returns an EventListeners struct that must be kept alive for the duration
/// of the window's lifetime. Dropping it will not remove the listeners (they
/// are attached to the DOM), but the closures will be invalidated.
#[cfg(target_arch = "wasm32")]
pub fn setup_event_listeners(
    canvas: &web_sys::HtmlCanvasElement,
    window: Rc<WebWindow>,
) -> Result<EventListeners, JsValue> {
    // Make canvas focusable for keyboard events
    canvas.set_tab_index(0);

    // Get performance object for timestamps
    let performance = web_sys::window()
        .and_then(|w| w.performance())
        .ok_or_else(|| JsValue::from_str("No performance API"))?;

    // Mouse down
    let window_mousedown = window.clone();
    let perf_mousedown = performance.clone();
    let mousedown = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
        event.prevent_default();
        let now = perf_mousedown.now();
        window_mousedown.handle_mouse_down(&event, now);
    });
    canvas.add_event_listener_with_callback("mousedown", mousedown.as_ref().unchecked_ref())?;

    // Mouse up
    let window_mouseup = window.clone();
    let mouseup = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
        event.prevent_default();
        window_mouseup.handle_mouse_up(&event);
    });
    canvas.add_event_listener_with_callback("mouseup", mouseup.as_ref().unchecked_ref())?;

    // Mouse move
    let window_mousemove = window.clone();
    let mousemove = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
        window_mousemove.handle_mouse_move_event(&event);
    });
    canvas.add_event_listener_with_callback("mousemove", mousemove.as_ref().unchecked_ref())?;

    // Mouse enter
    let window_mouseenter = window.clone();
    let mouseenter = Closure::<dyn FnMut(_)>::new(move |_event: web_sys::MouseEvent| {
        window_mouseenter.handle_mouse_enter();
    });
    canvas.add_event_listener_with_callback("mouseenter", mouseenter.as_ref().unchecked_ref())?;

    // Mouse leave
    let window_mouseleave = window.clone();
    let mouseleave = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
        window_mouseleave.handle_mouse_leave(&event);
    });
    canvas.add_event_listener_with_callback("mouseleave", mouseleave.as_ref().unchecked_ref())?;

    // Wheel (scroll)
    let window_wheel = window.clone();
    let wheel = Closure::<dyn FnMut(_)>::new(move |event: web_sys::WheelEvent| {
        event.prevent_default();
        window_wheel.handle_wheel(&event);
    });
    // Use passive: false to allow preventDefault on wheel
    let wheel_options = web_sys::AddEventListenerOptions::new();
    wheel_options.set_passive(false);
    canvas.add_event_listener_with_callback_and_add_event_listener_options(
        "wheel",
        wheel.as_ref().unchecked_ref(),
        &wheel_options,
    )?;

    // Key down - attach to canvas (needs focus)
    let window_keydown = window.clone();
    let keydown = Closure::<dyn FnMut(_)>::new(move |event: web_sys::KeyboardEvent| {
        // Don't prevent default for all keys - allow browser shortcuts
        // Only prevent for keys we're handling
        let key = event.key();
        if !should_allow_browser_default(&key) {
            event.prevent_default();
        }
        window_keydown.handle_key_down(&event);
    });
    canvas.add_event_listener_with_callback("keydown", keydown.as_ref().unchecked_ref())?;

    // Key up
    let window_keyup = window.clone();
    let keyup = Closure::<dyn FnMut(_)>::new(move |event: web_sys::KeyboardEvent| {
        window_keyup.handle_key_up(&event);
    });
    canvas.add_event_listener_with_callback("keyup", keyup.as_ref().unchecked_ref())?;

    // Focus
    let window_focus = window.clone();
    let focus = Closure::<dyn FnMut(_)>::new(move |_event: web_sys::FocusEvent| {
        window_focus.handle_focus();
    });
    canvas.add_event_listener_with_callback("focus", focus.as_ref().unchecked_ref())?;

    // Blur
    let window_blur = window.clone();
    let blur = Closure::<dyn FnMut(_)>::new(move |_event: web_sys::FocusEvent| {
        window_blur.handle_blur();
    });
    canvas.add_event_listener_with_callback("blur", blur.as_ref().unchecked_ref())?;

    // Window resize - attach to window, not canvas
    let window_resize = window.clone();
    let canvas_for_resize = canvas.clone();
    let resize = Closure::<dyn FnMut(_)>::new(move |_event: web_sys::Event| {
        let width = canvas_for_resize.client_width() as f32;
        let height = canvas_for_resize.client_height() as f32;
        window_resize.handle_resize(width, height);
    });
    if let Some(browser_window) = web_sys::window() {
        browser_window.add_event_listener_with_callback("resize", resize.as_ref().unchecked_ref())?;
    }

    // Context menu - prevent right-click menu
    let contextmenu = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
        event.prevent_default();
    });
    canvas.add_event_listener_with_callback("contextmenu", contextmenu.as_ref().unchecked_ref())?;
    contextmenu.forget(); // This one can be forgotten since it doesn't reference window

    Ok(EventListeners {
        _mousedown: mousedown,
        _mouseup: mouseup,
        _mousemove: mousemove,
        _mouseenter: mouseenter,
        _mouseleave: mouseleave,
        _wheel: wheel,
        _keydown: keydown,
        _keyup: keyup,
        _focus: focus,
        _blur: blur,
        _resize: resize,
    })
}

/// Check if a key should allow browser default behavior
#[cfg(target_arch = "wasm32")]
fn should_allow_browser_default(key: &str) -> bool {
    // Allow browser shortcuts like F5 (refresh), F11 (fullscreen), F12 (devtools)
    matches!(key, "F5" | "F11" | "F12")
}

/// Start the requestAnimationFrame render loop
///
/// This sets up a continuous animation loop that calls the window's
/// request_frame method on each frame.
#[cfg(target_arch = "wasm32")]
pub fn start_animation_loop(window: Rc<WebWindow>) -> Result<(), JsValue> {
    // Use a shared reference for the recursive closure
    // IMPORTANT: We use Rc<RefCell<Option<Closure>>> pattern to allow the closure
    // to reference itself for scheduling the next frame
    let callback: Rc<std::cell::RefCell<Option<Closure<dyn FnMut()>>>> =
        Rc::new(std::cell::RefCell::new(None));
    let callback_clone = callback.clone();

    let browser_window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;

    // Create the closure that will call itself recursively
    let closure = Closure::new(move || {
        // Request the frame from GPUI
        window.request_frame();

        // Schedule next frame using the cloned reference
        if let Some(browser_window) = web_sys::window() {
            if let Some(ref cb) = *callback_clone.borrow() {
                let _ = browser_window.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }
    });

    // Store the closure in the shared cell so the recursive reference works
    *callback.borrow_mut() = Some(closure);

    // Start the loop
    if let Some(ref cb) = *callback.borrow() {
        browser_window.request_animation_frame(cb.as_ref().unchecked_ref())?;
    }

    // Leak the Rc to keep the closure alive forever
    // (The closure is stored inside the RefCell, so leaking the Rc keeps it alive)
    std::mem::forget(callback);

    Ok(())
}

//=============================================================================
// Non-WASM stubs
//=============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub struct EventListeners;

#[cfg(not(target_arch = "wasm32"))]
pub fn setup_event_listeners(
    _canvas: &(),
    _window: std::rc::Rc<super::window::WebWindow>,
) -> Result<EventListeners, String> {
    Ok(EventListeners)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_animation_loop(_window: std::rc::Rc<super::window::WebWindow>) -> Result<(), String> {
    Ok(())
}
