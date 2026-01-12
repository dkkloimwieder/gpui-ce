//! Web renderer for GPUI using blade-graphics WebGPU backend
//!
//! This provides a simplified renderer for WASM that works with the single-threaded
//! browser environment. Unlike the native BladeRenderer, this doesn't require
//! Send+Sync bounds since WASM is inherently single-threaded.
//!
//! Note: GPU context initialization on WASM is async. Use `initialize_async` with
//! wasm-bindgen-futures to properly initialize the renderer.

use crate::{DevicePixels, Scene, Size, size};
use std::cell::RefCell;
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
use blade_graphics as gpu;

#[cfg(target_arch = "wasm32")]
use std::ptr;

/// Global parameters passed to all shaders.
///
/// This struct must match the layout in shaders.wgsl:
/// ```wgsl
/// struct GlobalParams {
///     viewport_size: vec2<f32>,
///     premultiplied_alpha: u32,
///     pad: u32,
/// }
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlobalParams {
    /// Viewport size in pixels
    pub viewport_size: [f32; 2],
    /// Whether to use premultiplied alpha (1) or not (0)
    pub premultiplied_alpha: u32,
    /// Padding for alignment
    pub pad: u32,
}

impl Default for GlobalParams {
    fn default() -> Self {
        Self {
            viewport_size: [800.0, 600.0],
            premultiplied_alpha: 0,
            pad: 0,
        }
    }
}

/// Configuration for the web renderer surface
pub struct WebSurfaceConfig {
    /// Size of the surface in device pixels
    pub size: gpu::Extent,
    /// Whether the surface should be transparent
    pub transparent: bool,
}

impl Default for WebSurfaceConfig {
    fn default() -> Self {
        Self {
            size: gpu::Extent {
                width: 800,
                height: 600,
                depth: 1,
            },
            transparent: false,
        }
    }
}

/// Web renderer state - not Send/Sync since WASM is single-threaded
pub struct WebRendererState {
    /// GPU context
    pub gpu: gpu::Context,
    /// Rendering surface
    pub surface: gpu::Surface,
    /// Surface configuration
    pub surface_config: gpu::SurfaceConfig,
    /// Command encoder
    pub command_encoder: gpu::CommandEncoder,
    /// Last sync point for frame pacing
    pub last_sync_point: Option<gpu::SyncPoint>,
    /// Current drawable size
    pub drawable_size: Size<DevicePixels>,
    /// Global parameters for shaders
    pub globals: GlobalParams,
    /// GPU buffer for global parameters
    pub globals_buffer: gpu::Buffer,
}

/// Web renderer for GPUI
///
/// This is wrapped in Rc<RefCell<>> since WASM is single-threaded
/// and we don't need Send+Sync.
pub struct WebRenderer(pub Rc<RefCell<Option<WebRendererState>>>);

impl WebRenderer {
    /// Create a new web renderer (uninitialized)
    ///
    /// Call `initialize` to set up the GPU context and surface.
    pub fn new() -> Self {
        WebRenderer(Rc::new(RefCell::new(None)))
    }

    /// Check if the renderer is initialized
    pub fn is_initialized(&self) -> bool {
        self.0.borrow().is_some()
    }

    /// Initialize the renderer with a canvas element (async)
    ///
    /// On WASM, GPU context initialization is async. This method must be
    /// called with wasm-bindgen-futures::spawn_local or similar.
    ///
    /// # Example
    /// ```ignore
    /// use wasm_bindgen_futures::spawn_local;
    ///
    /// let renderer = WebRenderer::new();
    /// let canvas = get_canvas_element();
    /// spawn_local(async move {
    ///     renderer.initialize_async(canvas, config).await.unwrap();
    /// });
    /// ```
    #[cfg(target_arch = "wasm32")]
    pub async fn initialize_async(
        &self,
        canvas: web_sys::HtmlCanvasElement,
        config: WebSurfaceConfig,
    ) -> anyhow::Result<()> {
        // Create GPU context asynchronously
        let gpu = gpu::Context::init_async(gpu::ContextDesc {
            presentation: true,
            validation: cfg!(debug_assertions),
            ..Default::default()
        })
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create GPU context: {:?}", e))?;

        // Create surface from canvas
        let mut surface = gpu
            .create_surface_from_canvas(canvas)
            .map_err(|e| anyhow::anyhow!("Failed to create surface from canvas: {:?}", e))?;

        // Configure the surface
        let surface_config = gpu::SurfaceConfig {
            size: config.size,
            usage: gpu::TextureUsage::TARGET,
            display_sync: gpu::DisplaySync::Recent,
            color_space: gpu::ColorSpace::Srgb,
            allow_exclusive_full_screen: false,
            transparent: config.transparent,
        };

        gpu.reconfigure_surface(&mut surface, surface_config.clone());

        // Create command encoder
        let command_encoder = gpu.create_command_encoder(gpu::CommandEncoderDesc {
            name: "web-renderer",
            buffer_count: 2,
        });

        let drawable_size = Size {
            width: DevicePixels(config.size.width as i32),
            height: DevicePixels(config.size.height as i32),
        };

        // Determine premultiplied alpha from surface info
        let premultiplied_alpha = match surface.info().alpha {
            gpu::AlphaMode::Ignored | gpu::AlphaMode::PostMultiplied => 0,
            gpu::AlphaMode::PreMultiplied => 1,
        };

        // Create global parameters
        let globals = GlobalParams {
            viewport_size: [config.size.width as f32, config.size.height as f32],
            premultiplied_alpha,
            pad: 0,
        };

        // Create GPU buffer for globals (uniform buffer)
        let globals_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "globals",
            size: std::mem::size_of::<GlobalParams>() as u64,
            memory: gpu::Memory::Shared,
        });

        // Initialize the buffer with current globals
        unsafe {
            ptr::copy_nonoverlapping(
                &globals as *const GlobalParams,
                globals_buffer.data() as *mut GlobalParams,
                1,
            );
        }
        gpu.sync_buffer(globals_buffer);

        *self.0.borrow_mut() = Some(WebRendererState {
            gpu,
            surface,
            surface_config,
            command_encoder,
            last_sync_point: None,
            drawable_size,
            globals,
            globals_buffer,
        });

        Ok(())
    }

    /// Update the drawable size (call on resize)
    #[cfg(target_arch = "wasm32")]
    pub fn update_drawable_size(&self, size: Size<DevicePixels>) {
        if let Some(state) = self.0.borrow_mut().as_mut() {
            state.drawable_size = size;
            state.surface_config.size = gpu::Extent {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth: 1,
            };
            state.gpu.reconfigure_surface(&mut state.surface, state.surface_config.clone());

            // Update globals with new viewport size
            state.globals.viewport_size = [size.width.0 as f32, size.height.0 as f32];
            unsafe {
                ptr::copy_nonoverlapping(
                    &state.globals as *const GlobalParams,
                    state.globals_buffer.data() as *mut GlobalParams,
                    1,
                );
            }
            state.gpu.sync_buffer(state.globals_buffer);
        }
    }

    /// Update the drawable size (call on resize) - non-WASM stub
    #[cfg(not(target_arch = "wasm32"))]
    pub fn update_drawable_size(&self, _size: Size<DevicePixels>) {
        // No-op on non-WASM
    }

    /// Draw a scene
    ///
    /// This is a simplified draw that just clears the screen.
    /// Full scene rendering will be implemented in follow-up work.
    #[cfg(target_arch = "wasm32")]
    pub fn draw(&self, _scene: &Scene) {
        let mut state_ref = self.0.borrow_mut();
        let Some(state) = state_ref.as_mut() else {
            log::warn!("WebRenderer::draw called before initialization");
            return;
        };

        // Wait for previous frame
        if let Some(ref sp) = state.last_sync_point {
            let _ = state.gpu.wait_for(sp, 1000);
        }

        // Acquire frame
        let frame = state.surface.acquire_frame();
        if !frame.is_valid() {
            log::warn!("Failed to acquire frame");
            return;
        }

        // Begin encoding
        state.command_encoder.start();

        // Get the texture view for rendering
        let target = frame.texture_view();

        // Render pass to clear the screen
        {
            let _pass = state.command_encoder.render("clear", gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            });

            // Scene rendering would go here
            // For now, the screen is just cleared
        }

        // Queue frame for presentation
        state.command_encoder.present(frame);

        // Submit
        let sync_point = state.gpu.submit(&mut state.command_encoder);
        state.last_sync_point = Some(sync_point);
    }

    /// Draw a scene (non-WASM stub)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn draw(&self, _scene: &Scene) {
        // No-op on non-WASM
    }

    /// Clear the screen to black
    ///
    /// This is useful for testing that WebGPU is working before full scene
    /// rendering is implemented.
    #[cfg(target_arch = "wasm32")]
    pub fn clear(&self) {
        let mut state_ref = self.0.borrow_mut();
        let Some(state) = state_ref.as_mut() else {
            log::warn!("WebRenderer::clear called before initialization");
            return;
        };

        // Wait for previous frame
        if let Some(ref sp) = state.last_sync_point {
            let _ = state.gpu.wait_for(sp, 1000);
        }

        // Acquire frame
        let frame = state.surface.acquire_frame();
        if !frame.is_valid() {
            log::warn!("Failed to acquire frame");
            return;
        }

        // Begin encoding
        state.command_encoder.start();

        // Get the texture view for rendering
        let target = frame.texture_view();

        // Render pass to clear the screen to opaque black
        {
            let _pass = state.command_encoder.render("clear", gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            });
        }

        // Queue frame for presentation
        state.command_encoder.present(frame);

        // Submit
        let sync_point = state.gpu.submit(&mut state.command_encoder);
        state.last_sync_point = Some(sync_point);
    }

    /// Clear the screen (non-WASM stub)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn clear(&self) {
        // No-op on non-WASM
    }

    /// Clear the screen with a color index (for testing color cycling)
    ///
    /// Index 0: Black, 1: White, 2: Transparent (shows HTML background), 3: Black again
    #[cfg(target_arch = "wasm32")]
    pub fn clear_with_index(&self, color_index: u32) {
        let color = match color_index % 3 {
            0 => gpu::TextureColor::OpaqueBlack,
            1 => gpu::TextureColor::White,
            _ => gpu::TextureColor::TransparentBlack,
        };

        let mut state_ref = self.0.borrow_mut();
        let Some(state) = state_ref.as_mut() else {
            log::warn!("WebRenderer::clear_with_index called before initialization");
            return;
        };

        // Wait for previous frame
        if let Some(ref sp) = state.last_sync_point {
            let _ = state.gpu.wait_for(sp, 1000);
        }

        // Acquire frame
        let frame = state.surface.acquire_frame();
        if !frame.is_valid() {
            log::warn!("Failed to acquire frame");
            return;
        }

        // Begin encoding
        state.command_encoder.start();

        // Get the texture view for rendering
        let target = frame.texture_view();

        // Render pass to clear the screen
        {
            let _pass = state.command_encoder.render("clear", gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target,
                    init_op: gpu::InitOp::Clear(color),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            });
        }

        // Queue frame for presentation
        state.command_encoder.present(frame);

        // Submit
        let sync_point = state.gpu.submit(&mut state.command_encoder);
        state.last_sync_point = Some(sync_point);
    }

    /// Clear the screen with color index (non-WASM stub)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn clear_with_index(&self, _color_index: u32) {
        // No-op on non-WASM
    }

    /// Get the current drawable size
    pub fn drawable_size(&self) -> Size<DevicePixels> {
        self.0
            .borrow()
            .as_ref()
            .map(|s| s.drawable_size)
            .unwrap_or_else(|| size(DevicePixels(800), DevicePixels(600)))
    }

    /// Get the current global parameters
    pub fn globals(&self) -> GlobalParams {
        self.0
            .borrow()
            .as_ref()
            .map(|s| s.globals)
            .unwrap_or_default()
    }

    /// Get a buffer piece for the globals buffer (for shader binding)
    #[cfg(target_arch = "wasm32")]
    pub fn globals_buffer_piece(&self) -> Option<gpu::BufferPiece> {
        self.0.borrow().as_ref().map(|s| gpu::BufferPiece {
            buffer: s.globals_buffer,
            offset: 0,
        })
    }
}

impl Default for WebRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholder for non-WASM builds
#[cfg(not(target_arch = "wasm32"))]
impl WebRenderer {
    pub fn initialize(&self, _config: WebSurfaceConfig) -> anyhow::Result<()> {
        Ok(())
    }
}
