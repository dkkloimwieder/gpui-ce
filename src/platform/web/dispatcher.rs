//! Web dispatcher for GPUI
//!
//! Handles task scheduling using browser APIs (setTimeout, requestAnimationFrame).

use crate::{
    PlatformDispatcher, Priority, RealtimePriority, RunnableVariant,
    TaskLabel, TaskTiming, ThreadTaskTimings,
};
use parking_lot::Mutex;
use std::{
    cell::RefCell,
    collections::VecDeque,
    sync::Arc,
    time::Duration,
};

// Thread-local storage for the global dispatcher reference
// This allows the animation loop to poll pending tasks
thread_local! {
    static GLOBAL_DISPATCHER: RefCell<Option<Arc<WebDispatcher>>> = const { RefCell::new(None) };
}

/// Set the global dispatcher reference (called when platform is created)
pub fn set_global_dispatcher(dispatcher: Arc<WebDispatcher>) {
    GLOBAL_DISPATCHER.with(|d| {
        *d.borrow_mut() = Some(dispatcher);
    });
}

/// Poll the global dispatcher to run pending tasks
/// Called from the requestAnimationFrame loop
pub fn poll_global_dispatcher() {
    GLOBAL_DISPATCHER.with(|d| {
        if let Some(ref dispatcher) = *d.borrow() {
            dispatcher.poll();
        }
    });
}

/// Web dispatcher - schedules tasks using browser APIs
pub struct WebDispatcher {
    /// Queue of tasks to run on main thread
    main_thread_tasks: Arc<Mutex<VecDeque<RunnableVariant>>>,
    /// Whether we're on the main thread (always true for WASM)
    is_main_thread: bool,
}

impl WebDispatcher {
    /// Create a new web dispatcher
    pub fn new() -> Self {
        Self {
            main_thread_tasks: Arc::new(Mutex::new(VecDeque::new())),
            is_main_thread: true,
        }
    }

    /// Poll and run pending tasks
    /// Called from requestAnimationFrame loop
    pub fn poll(&self) {
        // Run all pending main thread tasks
        loop {
            let task = self.main_thread_tasks.lock().pop_front();
            match task {
                Some(RunnableVariant::Meta(runnable)) => {
                    runnable.run();
                }
                Some(RunnableVariant::Compat(runnable)) => {
                    runnable.run();
                }
                None => break,
            }
        }
    }
}

impl PlatformDispatcher for WebDispatcher {
    fn get_all_timings(&self) -> Vec<ThreadTaskTimings> {
        Vec::new()
    }

    fn get_current_thread_timings(&self) -> Vec<TaskTiming> {
        Vec::new()
    }

    fn is_main_thread(&self) -> bool {
        // WASM is single-threaded, always on main thread
        true
    }

    fn dispatch(&self, runnable: RunnableVariant, _label: Option<TaskLabel>, _priority: Priority) {
        // On WASM, all tasks run on the main thread
        self.main_thread_tasks.lock().push_back(runnable);
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, _priority: Priority) {
        self.main_thread_tasks.lock().push_back(runnable);
    }

    fn dispatch_after(&self, duration: Duration, runnable: RunnableVariant) {
        // Use setTimeout via wasm-bindgen
        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::prelude::*;
            use wasm_bindgen::JsCast;

            let tasks = self.main_thread_tasks.clone();
            let millis = duration.as_millis() as i32;

            let closure = Closure::once(Box::new(move || {
                tasks.lock().push_back(runnable);
            }) as Box<dyn FnOnce()>);

            if let Some(window) = web_sys::window() {
                let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                    closure.as_ref().unchecked_ref(),
                    millis,
                );
            }

            // Prevent closure from being dropped
            closure.forget();
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // On native (for testing), just queue immediately
            let _ = duration;
            self.main_thread_tasks.lock().push_back(runnable);
        }
    }

    fn spawn_realtime(&self, _priority: RealtimePriority, f: Box<dyn FnOnce() + Send>) {
        // WASM is single-threaded, just run on main thread
        // Queue it to run in the next poll
        #[cfg(target_arch = "wasm32")]
        {
            // Can't easily wrap FnOnce in RunnableVariant, so run immediately
            f();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            f();
        }
    }
}
