//! Web renderer for GPUI using blade-graphics WebGPU backend
//!
//! This provides a simplified renderer for WASM that works with the single-threaded
//! browser environment. Unlike the native BladeRenderer, this doesn't require
//! Send+Sync bounds since WASM is inherently single-threaded.
//!
//! Note: GPU context initialization on WASM is async. Use `initialize_async` with
//! wasm-bindgen-futures to properly initialize the renderer.

use crate::{DevicePixels, PlatformAtlas, Scene, Size, size};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use blade_graphics as gpu;

#[cfg(target_arch = "wasm32")]
use std::{mem, ptr};

#[cfg(target_arch = "wasm32")]
use crate::scene::{Quad, MonochromeSprite, PolychromeSprite};

#[cfg(target_arch = "wasm32")]
use super::web_atlas::WebGpuAtlas;

/// Shader data layout for quad rendering
#[cfg(target_arch = "wasm32")]
#[derive(blade_macros::ShaderData)]
struct ShaderQuadsData {
    globals: GlobalParams,
    b_quads: gpu::BufferPiece,
}

/// Shader data layout for monochrome sprite rendering
#[cfg(target_arch = "wasm32")]
#[derive(blade_macros::ShaderData)]
struct ShaderMonoSpritesData {
    globals: GlobalParams,
    t_sprite: gpu::TextureView,
    s_sprite: gpu::Sampler,
    b_mono_sprites: gpu::BufferPiece,
}

/// Shader data layout for polychrome sprite rendering
#[cfg(target_arch = "wasm32")]
#[derive(blade_macros::ShaderData)]
struct ShaderPolySpritesData {
    globals: GlobalParams,
    t_sprite: gpu::TextureView,
    s_sprite: gpu::Sampler,
    b_poly_sprites: gpu::BufferPiece,
}

/// Maximum number of quads per batch
#[cfg(target_arch = "wasm32")]
const MAX_QUADS_PER_BATCH: usize = 4096;

/// Maximum number of sprites per batch
#[cfg(target_arch = "wasm32")]
const MAX_SPRITES_PER_BATCH: usize = 4096;

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
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
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
    /// GPU context (shared via Rc for atlas)
    pub gpu: Rc<gpu::Context>,
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
    /// Quad render pipeline
    pub quad_pipeline: gpu::RenderPipeline,
    /// Buffer for quad instance data
    pub quad_buffer: gpu::Buffer,
    /// Monochrome sprite render pipeline
    pub mono_sprite_pipeline: gpu::RenderPipeline,
    /// Buffer for monochrome sprite instance data
    pub mono_sprite_buffer: gpu::Buffer,
    /// Polychrome sprite render pipeline
    pub poly_sprite_pipeline: gpu::RenderPipeline,
    /// Buffer for polychrome sprite instance data
    pub poly_sprite_buffer: gpu::Buffer,
    /// Sampler for atlas textures
    pub atlas_sampler: gpu::Sampler,
    /// Texture atlas for sprites/glyphs (Arc for sharing with window)
    pub atlas: Arc<WebGpuAtlas>,
}

/// Web renderer for GPUI
///
/// This is wrapped in Rc<RefCell<>> since WASM is single-threaded
/// and we don't need Send+Sync.
#[derive(Clone)]
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
        // Create GPU context asynchronously (wrapped in Rc for sharing with atlas)
        let gpu = Rc::new(gpu::Context::init_async(gpu::ContextDesc {
            presentation: true,
            validation: cfg!(debug_assertions),
            ..Default::default()
        })
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create GPU context: {:?}", e))?);

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

        // Create shader module
        let shader_source = include_str!("shaders.wgsl");
        let shader = gpu.create_shader(gpu::ShaderDesc {
            source: shader_source,
        });

        // Create quad render pipeline
        let quad_layout = <ShaderQuadsData as gpu::ShaderData>::layout();
        let quad_pipeline = gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "quads",
            data_layouts: &[&quad_layout],
            vertex: shader.at("vs_quad"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_quad")),
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::ALL,
            }],
            multisample_state: gpu::MultisampleState::default(),
        });

        // Create quad instance buffer
        let quad_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "quads",
            size: (mem::size_of::<Quad>() * MAX_QUADS_PER_BATCH) as u64,
            memory: gpu::Memory::Shared,
        });

        // Create atlas sampler for sprite rendering
        let atlas_sampler = gpu.create_sampler(gpu::SamplerDesc {
            name: "atlas",
            address_modes: [gpu::AddressMode::ClampToEdge; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            mipmap_filter: gpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create monochrome sprite render pipeline
        let mono_sprite_layout = <ShaderMonoSpritesData as gpu::ShaderData>::layout();
        let mono_sprite_pipeline = gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "mono_sprites",
            data_layouts: &[&mono_sprite_layout],
            vertex: shader.at("vs_mono_sprite"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_mono_sprite")),
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::ALL,
            }],
            multisample_state: gpu::MultisampleState::default(),
        });

        // Create monochrome sprite instance buffer
        let mono_sprite_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "mono_sprites",
            size: (mem::size_of::<MonochromeSprite>() * MAX_SPRITES_PER_BATCH) as u64,
            memory: gpu::Memory::Shared,
        });

        // Create polychrome sprite render pipeline
        let poly_sprite_layout = <ShaderPolySpritesData as gpu::ShaderData>::layout();
        let poly_sprite_pipeline = gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "poly_sprites",
            data_layouts: &[&poly_sprite_layout],
            vertex: shader.at("vs_poly_sprite"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_poly_sprite")),
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::ALL,
            }],
            multisample_state: gpu::MultisampleState::default(),
        });

        // Create polychrome sprite instance buffer
        let poly_sprite_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "poly_sprites",
            size: (mem::size_of::<PolychromeSprite>() * MAX_SPRITES_PER_BATCH) as u64,
            memory: gpu::Memory::Shared,
        });

        // Create texture atlas for sprites and glyphs (Arc for sharing with window)
        let atlas = Arc::new(WebGpuAtlas::new(&gpu));

        *self.0.borrow_mut() = Some(WebRendererState {
            gpu,
            surface,
            surface_config,
            command_encoder,
            last_sync_point: None,
            drawable_size,
            globals,
            globals_buffer,
            quad_pipeline,
            quad_buffer,
            mono_sprite_pipeline,
            mono_sprite_buffer,
            poly_sprite_pipeline,
            poly_sprite_buffer,
            atlas_sampler,
            atlas,
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
    /// Renders all primitives from the scene including quads, shadows, etc.
    #[cfg(target_arch = "wasm32")]
    pub fn draw(&self, scene: &Scene) {
        use crate::PrimitiveBatch;

        let mut state_ref = self.0.borrow_mut();
        let Some(state) = state_ref.as_mut() else {
            log::warn!("WebRenderer::draw called before initialization");
            return;
        };

        // Wait for previous frame
        if let Some(ref sp) = state.last_sync_point {
            let _ = state.gpu.wait_for(sp, 1000);
        }

        // Flush any pending atlas uploads
        state.atlas.flush_uploads();

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

        // Main render pass
        {
            let mut pass = state.command_encoder.render("main", gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            });

            // Process batches from scene
            for batch in scene.batches() {
                match batch {
                    PrimitiveBatch::Quads(quads) => {
                        Self::draw_quads_internal(
                            &mut pass,
                            quads,
                            &state.globals,
                            state.globals_buffer,
                            state.quad_buffer,
                            &state.quad_pipeline,
                            &state.gpu,
                        );
                    }
                    PrimitiveBatch::MonochromeSprites { texture_id, sprites } => {
                        if let Some(tex_info) = state.atlas.get_texture_info(texture_id) {
                            Self::draw_mono_sprites_internal(
                                &mut pass,
                                sprites,
                                &state.globals,
                                tex_info.view,
                                state.atlas_sampler,
                                state.mono_sprite_buffer,
                                &state.mono_sprite_pipeline,
                                &state.gpu,
                            );
                        }
                    }
                    PrimitiveBatch::PolychromeSprites { texture_id, sprites } => {
                        if let Some(tex_info) = state.atlas.get_texture_info(texture_id) {
                            Self::draw_poly_sprites_internal(
                                &mut pass,
                                sprites,
                                &state.globals,
                                tex_info.view,
                                state.atlas_sampler,
                                state.poly_sprite_buffer,
                                &state.poly_sprite_pipeline,
                                &state.gpu,
                            );
                        }
                    }
                    // TODO: Other primitive types (shadows, paths, underlines, surfaces)
                    _ => {}
                }
            }
        }

        // Queue frame for presentation
        state.command_encoder.present(frame);

        // Submit
        let sync_point = state.gpu.submit(&mut state.command_encoder);
        state.last_sync_point = Some(sync_point);
    }

    /// Internal helper to draw quads during a render pass
    #[cfg(target_arch = "wasm32")]
    fn draw_quads_internal(
        pass: &mut gpu::RenderCommandEncoder,
        quads: &[Quad],
        globals: &GlobalParams,
        globals_buffer: gpu::Buffer,
        quad_buffer: gpu::Buffer,
        pipeline: &gpu::RenderPipeline,
        gpu: &gpu::Context,
    ) {
        if quads.is_empty() {
            return;
        }

        let count = quads.len().min(MAX_QUADS_PER_BATCH);

        // Upload quad data to buffer
        unsafe {
            ptr::copy_nonoverlapping(
                quads.as_ptr(),
                quad_buffer.data() as *mut Quad,
                count,
            );
        }
        gpu.sync_buffer(quad_buffer);

        // Bind pipeline and data
        let mut encoder = pass.with(pipeline);
        encoder.bind(
            0,
            &ShaderQuadsData {
                globals: *globals,
                b_quads: gpu::BufferPiece {
                    buffer: quad_buffer,
                    offset: 0,
                },
            },
        );

        // Draw instanced quads (4 vertices per quad, N instances)
        encoder.draw(0, 4, 0, count as u32);
    }

    /// Internal helper to draw monochrome sprites during a render pass
    #[cfg(target_arch = "wasm32")]
    fn draw_mono_sprites_internal(
        pass: &mut gpu::RenderCommandEncoder,
        sprites: &[MonochromeSprite],
        globals: &GlobalParams,
        texture_view: gpu::TextureView,
        sampler: gpu::Sampler,
        sprite_buffer: gpu::Buffer,
        pipeline: &gpu::RenderPipeline,
        gpu: &Rc<gpu::Context>,
    ) {
        if sprites.is_empty() {
            return;
        }

        let count = sprites.len().min(MAX_SPRITES_PER_BATCH);

        // Upload sprite data to buffer
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr(),
                sprite_buffer.data() as *mut MonochromeSprite,
                count,
            );
        }
        gpu.sync_buffer(sprite_buffer);

        // Bind pipeline and data
        let mut encoder = pass.with(pipeline);
        encoder.bind(
            0,
            &ShaderMonoSpritesData {
                globals: *globals,
                t_sprite: texture_view,
                s_sprite: sampler,
                b_mono_sprites: gpu::BufferPiece {
                    buffer: sprite_buffer,
                    offset: 0,
                },
            },
        );

        // Draw instanced sprites (4 vertices per sprite, N instances)
        encoder.draw(0, 4, 0, count as u32);
    }

    /// Internal helper to draw polychrome sprites during a render pass
    #[cfg(target_arch = "wasm32")]
    fn draw_poly_sprites_internal(
        pass: &mut gpu::RenderCommandEncoder,
        sprites: &[PolychromeSprite],
        globals: &GlobalParams,
        texture_view: gpu::TextureView,
        sampler: gpu::Sampler,
        sprite_buffer: gpu::Buffer,
        pipeline: &gpu::RenderPipeline,
        gpu: &Rc<gpu::Context>,
    ) {
        if sprites.is_empty() {
            return;
        }

        let count = sprites.len().min(MAX_SPRITES_PER_BATCH);

        // Upload sprite data to buffer
        unsafe {
            ptr::copy_nonoverlapping(
                sprites.as_ptr(),
                sprite_buffer.data() as *mut PolychromeSprite,
                count,
            );
        }
        gpu.sync_buffer(sprite_buffer);

        // Bind pipeline and data
        let mut encoder = pass.with(pipeline);
        encoder.bind(
            0,
            &ShaderPolySpritesData {
                globals: *globals,
                t_sprite: texture_view,
                s_sprite: sampler,
                b_poly_sprites: gpu::BufferPiece {
                    buffer: sprite_buffer,
                    offset: 0,
                },
            },
        );

        // Draw instanced sprites (4 vertices per sprite, N instances)
        encoder.draw(0, 4, 0, count as u32);
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

    /// Draw a test quad with specified bounds and color
    ///
    /// This is useful for testing that the quad shader is working.
    /// x, y, w, h are in pixels from top-left.
    /// Color is specified as (h, s, l, a) where h is 0-1, s is 0-1, l is 0-1, a is 0-1.
    #[cfg(target_arch = "wasm32")]
    pub fn draw_test_quad(&self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        use crate::{Background, Bounds, ContentMask, Corners, Edges, Hsla, ScaledPixels};
        use crate::scene::{BorderStyle, DrawOrder};

        let mut state_ref = self.0.borrow_mut();
        let Some(state) = state_ref.as_mut() else {
            log::warn!("WebRenderer::draw_test_quad called before initialization");
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

        // Create a test quad
        let quad = Quad {
            order: DrawOrder::default(),
            border_style: BorderStyle::default(),
            bounds: Bounds {
                origin: crate::Point {
                    x: ScaledPixels(x),
                    y: ScaledPixels(y),
                },
                size: crate::Size {
                    width: ScaledPixels(w),
                    height: ScaledPixels(h),
                },
            },
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: crate::Point {
                        x: ScaledPixels(0.0),
                        y: ScaledPixels(0.0),
                    },
                    size: crate::Size {
                        width: ScaledPixels(state.globals.viewport_size[0]),
                        height: ScaledPixels(state.globals.viewport_size[1]),
                    },
                },
            },
            background: Hsla {
                h: color[0],
                s: color[1],
                l: color[2],
                a: color[3],
            }.into(),
            border_color: Hsla::default(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        };

        // Begin encoding
        state.command_encoder.start();

        // Get the texture view for rendering
        let target = frame.texture_view();

        // Main render pass
        {
            let mut pass = state.command_encoder.render("main", gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: target,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            });

            Self::draw_quads_internal(
                &mut pass,
                &[quad],
                &state.globals,
                state.globals_buffer,
                state.quad_buffer,
                &state.quad_pipeline,
                &state.gpu,
            );
        }

        // Queue frame for presentation
        state.command_encoder.present(frame);

        // Submit
        let sync_point = state.gpu.submit(&mut state.command_encoder);
        state.last_sync_point = Some(sync_point);
    }

    /// Draw a test quad (non-WASM stub)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn draw_test_quad(&self, _x: f32, _y: f32, _w: f32, _h: f32, _color: [f32; 4]) {
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

    /// Get access to the texture atlas for sprite/glyph rendering
    ///
    /// Returns None if the renderer hasn't been initialized yet.
    /// This returns Arc<dyn PlatformAtlas> for use with GPUI's sprite_atlas() method.
    #[cfg(target_arch = "wasm32")]
    pub fn sprite_atlas(&self) -> Option<Arc<dyn PlatformAtlas>> {
        self.0.borrow().as_ref().map(|s| s.atlas.clone() as Arc<dyn PlatformAtlas>)
    }

    /// Get the raw atlas reference (for internal use)
    #[cfg(target_arch = "wasm32")]
    pub fn atlas(&self) -> Option<Arc<WebGpuAtlas>> {
        self.0.borrow().as_ref().map(|s| s.atlas.clone())
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
