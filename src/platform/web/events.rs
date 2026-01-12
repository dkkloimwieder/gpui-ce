//! Browser event handling for GPUI
//!
//! This module converts browser DOM events to GPUI's PlatformInput types.

use crate::{
    KeyDownEvent, KeyUpEvent, Keystroke, Modifiers, ModifiersChangedEvent, MouseButton,
    MouseDownEvent, MouseExitEvent, MouseMoveEvent, MouseUpEvent, NavigationDirection,
    ScrollDelta, ScrollWheelEvent, TouchPhase, point, px,
};

/// Extract GPUI Modifiers from a browser MouseEvent
#[cfg(target_arch = "wasm32")]
pub fn modifiers_from_mouse_event(event: &web_sys::MouseEvent) -> Modifiers {
    Modifiers {
        control: event.ctrl_key(),
        alt: event.alt_key(),
        shift: event.shift_key(),
        platform: event.meta_key(),
        function: false, // Not available in browser
    }
}

/// Extract GPUI Modifiers from a browser KeyboardEvent
#[cfg(target_arch = "wasm32")]
pub fn modifiers_from_keyboard_event(event: &web_sys::KeyboardEvent) -> Modifiers {
    Modifiers {
        control: event.ctrl_key(),
        alt: event.alt_key(),
        shift: event.shift_key(),
        platform: event.meta_key(),
        function: false, // Not available in browser
    }
}

/// Convert browser mouse button number to GPUI MouseButton
#[cfg(target_arch = "wasm32")]
pub fn mouse_button_from_browser(button: i16) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        3 => MouseButton::Navigate(NavigationDirection::Back),
        4 => MouseButton::Navigate(NavigationDirection::Forward),
        _ => MouseButton::Left,
    }
}

/// Get the currently pressed mouse button from buttons bitmask
#[cfg(target_arch = "wasm32")]
pub fn pressed_button_from_buttons(buttons: u16) -> Option<MouseButton> {
    if buttons & 1 != 0 {
        Some(MouseButton::Left)
    } else if buttons & 2 != 0 {
        Some(MouseButton::Right)
    } else if buttons & 4 != 0 {
        Some(MouseButton::Middle)
    } else if buttons & 8 != 0 {
        Some(MouseButton::Navigate(NavigationDirection::Back))
    } else if buttons & 16 != 0 {
        Some(MouseButton::Navigate(NavigationDirection::Forward))
    } else {
        None
    }
}

/// Convert browser MouseEvent to GPUI MouseDownEvent
#[cfg(target_arch = "wasm32")]
pub fn mouse_down_from_browser(event: &web_sys::MouseEvent, click_count: usize) -> MouseDownEvent {
    MouseDownEvent {
        button: mouse_button_from_browser(event.button()),
        position: point(px(event.offset_x() as f32), px(event.offset_y() as f32)),
        modifiers: modifiers_from_mouse_event(event),
        click_count,
        first_mouse: false, // Browser windows are always focused on click
    }
}

/// Convert browser MouseEvent to GPUI MouseUpEvent
#[cfg(target_arch = "wasm32")]
pub fn mouse_up_from_browser(event: &web_sys::MouseEvent, click_count: usize) -> MouseUpEvent {
    MouseUpEvent {
        button: mouse_button_from_browser(event.button()),
        position: point(px(event.offset_x() as f32), px(event.offset_y() as f32)),
        modifiers: modifiers_from_mouse_event(event),
        click_count,
    }
}

/// Convert browser MouseEvent to GPUI MouseMoveEvent
#[cfg(target_arch = "wasm32")]
pub fn mouse_move_from_browser(event: &web_sys::MouseEvent) -> MouseMoveEvent {
    MouseMoveEvent {
        position: point(px(event.offset_x() as f32), px(event.offset_y() as f32)),
        pressed_button: pressed_button_from_buttons(event.buttons()),
        modifiers: modifiers_from_mouse_event(event),
    }
}

/// Convert browser MouseEvent to GPUI MouseExitEvent
#[cfg(target_arch = "wasm32")]
pub fn mouse_exit_from_browser(event: &web_sys::MouseEvent) -> MouseExitEvent {
    MouseExitEvent {
        position: point(px(event.offset_x() as f32), px(event.offset_y() as f32)),
        pressed_button: pressed_button_from_buttons(event.buttons()),
        modifiers: modifiers_from_mouse_event(event),
    }
}

/// Convert browser WheelEvent to GPUI ScrollWheelEvent
#[cfg(target_arch = "wasm32")]
pub fn scroll_wheel_from_browser(event: &web_sys::WheelEvent) -> ScrollWheelEvent {
    // Cast to MouseEvent to get position/modifiers (WheelEvent extends MouseEvent)
    let mouse_event: &web_sys::MouseEvent = event.as_ref();

    // WheelEvent.deltaMode: 0 = pixels, 1 = lines, 2 = pages
    let delta = match event.delta_mode() {
        0 => ScrollDelta::Pixels(point(
            px(event.delta_x() as f32),
            px(event.delta_y() as f32),
        )),
        1 => ScrollDelta::Lines(point(event.delta_x() as f32, event.delta_y() as f32)),
        2 => {
            // Page mode - treat as lines with larger multiplier
            ScrollDelta::Lines(point(
                event.delta_x() as f32 * 20.0,
                event.delta_y() as f32 * 20.0,
            ))
        }
        _ => ScrollDelta::Lines(point(event.delta_x() as f32, event.delta_y() as f32)),
    };

    ScrollWheelEvent {
        position: point(
            px(mouse_event.offset_x() as f32),
            px(mouse_event.offset_y() as f32),
        ),
        delta,
        modifiers: modifiers_from_mouse_event(mouse_event),
        touch_phase: TouchPhase::Moved,
    }
}

/// Convert browser key code to GPUI key name
#[cfg(target_arch = "wasm32")]
pub fn key_from_browser(event: &web_sys::KeyboardEvent) -> String {
    let key = event.key();

    // Map browser key names to GPUI key names
    match key.as_str() {
        // Special keys
        " " => "space".to_string(),
        "Backspace" => "backspace".to_string(),
        "Tab" => "tab".to_string(),
        "Enter" => "enter".to_string(),
        "Escape" => "escape".to_string(),
        "Delete" => "delete".to_string(),
        "Insert" => "insert".to_string(),

        // Arrow keys
        "ArrowUp" => "up".to_string(),
        "ArrowDown" => "down".to_string(),
        "ArrowLeft" => "left".to_string(),
        "ArrowRight" => "right".to_string(),

        // Navigation
        "Home" => "home".to_string(),
        "End" => "end".to_string(),
        "PageUp" => "pageup".to_string(),
        "PageDown" => "pagedown".to_string(),

        // Function keys
        "F1" => "f1".to_string(),
        "F2" => "f2".to_string(),
        "F3" => "f3".to_string(),
        "F4" => "f4".to_string(),
        "F5" => "f5".to_string(),
        "F6" => "f6".to_string(),
        "F7" => "f7".to_string(),
        "F8" => "f8".to_string(),
        "F9" => "f9".to_string(),
        "F10" => "f10".to_string(),
        "F11" => "f11".to_string(),
        "F12" => "f12".to_string(),

        // Modifier keys (when pressed as keys)
        "Control" => "control".to_string(),
        "Alt" => "alt".to_string(),
        "Shift" => "shift".to_string(),
        "Meta" => "cmd".to_string(),

        // Regular keys - use lowercase
        _ => {
            if key.len() == 1 {
                key.to_lowercase()
            } else {
                key.to_lowercase()
            }
        }
    }
}

/// Convert browser KeyboardEvent to GPUI KeyDownEvent
#[cfg(target_arch = "wasm32")]
pub fn key_down_from_browser(event: &web_sys::KeyboardEvent) -> KeyDownEvent {
    let key = key_from_browser(event);
    let modifiers = modifiers_from_keyboard_event(event);

    // For printable characters, set key_char
    let key_char = if event.key().len() == 1 && !modifiers.control && !modifiers.platform {
        Some(event.key())
    } else {
        None
    };

    KeyDownEvent {
        keystroke: Keystroke {
            modifiers,
            key,
            key_char,
        },
        is_held: event.repeat(),
        prefer_character_input: false,
    }
}

/// Convert browser KeyboardEvent to GPUI KeyUpEvent
#[cfg(target_arch = "wasm32")]
pub fn key_up_from_browser(event: &web_sys::KeyboardEvent) -> KeyUpEvent {
    let key = key_from_browser(event);
    let modifiers = modifiers_from_keyboard_event(event);

    KeyUpEvent {
        keystroke: Keystroke {
            modifiers,
            key,
            key_char: None,
        },
    }
}

/// Create a ModifiersChangedEvent from current modifier state
#[cfg(target_arch = "wasm32")]
pub fn modifiers_changed_from_keyboard(event: &web_sys::KeyboardEvent) -> ModifiersChangedEvent {
    ModifiersChangedEvent {
        modifiers: modifiers_from_keyboard_event(event),
        capslock: crate::Capslock {
            on: event.get_modifier_state("CapsLock"),
        },
    }
}

/// Check if a keyboard event is a modifier key press
#[cfg(target_arch = "wasm32")]
pub fn is_modifier_key(event: &web_sys::KeyboardEvent) -> bool {
    matches!(
        event.key().as_str(),
        "Control" | "Alt" | "Shift" | "Meta" | "CapsLock"
    )
}

//=============================================================================
// Non-WASM stubs for compilation
//=============================================================================

#[cfg(not(target_arch = "wasm32"))]
pub fn modifiers_from_mouse_event(_event: &()) -> Modifiers {
    Modifiers::default()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn modifiers_from_keyboard_event(_event: &()) -> Modifiers {
    Modifiers::default()
}
