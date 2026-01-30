//! Web renderer for GPUI using blade-graphics WebGPU backend
//!
//! This provides a simplified renderer for WASM that works with the single-threaded
//! browser environment. Unlike the native BladeRenderer, this doesn't require
//! Send+Sync bounds since WASM is inherently single-threaded.
//!
//! Note: GPU context initialization on WASM is async. Use `initialize_async` with
//! wasm-bindgen-futures to properly initialize the renderer.

use crate::{DevicePixels, PlatformAtlas, Scene, Size, size};
use crate::util::measure;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
use blade_graphics as gpu;

#[cfg(target_arch = "wasm32")]
use std::{mem, ptr};

#[cfg(target_arch = "wasm32")]
use crate::scene::{Quad, MonochromeSprite, PolychromeSprite, Shadow, Path, Underline};

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

/// Shader data layout for shadow rendering
#[cfg(target_arch = "wasm32")]
#[derive(blade_macros::ShaderData)]
struct ShaderShadowsData {
    globals: GlobalParams,
    b_shadows: gpu::BufferPiece,
}

/// Shader data layout for path rendering
#[cfg(target_arch = "wasm32")]
#[derive(blade_macros::ShaderData)]
struct ShaderPathsData {
    globals: GlobalParams,
    b_path_vertices: gpu::BufferPiece,
}

/// Shader data layout for underline rendering
#[cfg(target_arch = "wasm32")]
#[derive(blade_macros::ShaderData)]
struct ShaderUnderlinesData {
    globals: GlobalParams,
    b_underlines: gpu::BufferPiece,
}

/// GPU-side path vertex structure with full Background support.
/// Must match PathVertex in shaders.wgsl exactly.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct GpuPathVertex {
    xy_position_x: f32,
    xy_position_y: f32,
    st_position_x: f32,
    st_position_y: f32,
    // content_mask as Bounds (4 f32s)
    content_mask_origin_x: f32,
    content_mask_origin_y: f32,
    content_mask_size_width: f32,
    content_mask_size_height: f32,
    // path bounds for gradient calculation (4 f32s)
    bounds_origin_x: f32,
    bounds_origin_y: f32,
    bounds_size_width: f32,
    bounds_size_height: f32,
    // Background struct fields
    background_tag: u32,
    background_color_space: u32,
    // solid color (4 f32s)
    solid_h: f32,
    solid_s: f32,
    solid_l: f32,
    solid_a: f32,
    // gradient angle
    gradient_angle: f32,
    // color stop 0 (5 f32s: Hsla + percentage)
    stop0_h: f32,
    stop0_s: f32,
    stop0_l: f32,
    stop0_a: f32,
    stop0_percentage: f32,
    // color stop 1 (5 f32s: Hsla + percentage)
    stop1_h: f32,
    stop1_s: f32,
    stop1_l: f32,
    stop1_a: f32,
    stop1_percentage: f32,
    // padding to align to 16-byte boundary
    _pad: u32,
}

/// Maximum number of quads per batch
#[cfg(target_arch = "wasm32")]
const MAX_QUADS_PER_BATCH: usize = 4096;

/// Maximum number of sprites per batch
#[cfg(target_arch = "wasm32")]
const MAX_SPRITES_PER_BATCH: usize = 4096;

/// Maximum number of shadows per batch
#[cfg(target_arch = "wasm32")]
const MAX_SHADOWS_PER_BATCH: usize = 4096;

/// Maximum number of path vertices per batch
#[cfg(target_arch = "wasm32")]
const MAX_PATH_VERTICES_PER_BATCH: usize = 65536;

/// Maximum number of underlines per batch
#[cfg(target_arch = "wasm32")]
const MAX_UNDERLINES_PER_BATCH: usize = 4096;

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

/// MSAA sample count for antialiasing (4x MSAA)
const MSAA_SAMPLE_COUNT: u32 = 4;

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
    /// MSAA render target texture
    pub msaa_texture: gpu::Texture,
    /// MSAA render target view
    pub msaa_view: gpu::TextureView,
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
    /// Shadow render pipeline
    pub shadow_pipeline: gpu::RenderPipeline,
    /// Buffer for shadow instance data
    pub shadow_buffer: gpu::Buffer,
    /// Path render pipeline
    pub path_pipeline: gpu::RenderPipeline,
    /// Buffer for path vertex data
    pub path_buffer: gpu::Buffer,
    /// Underline render pipeline (straight)
    pub underline_pipeline: gpu::RenderPipeline,
    /// Underline render pipeline (wavy)
    pub underline_wavy_pipeline: gpu::RenderPipeline,
    /// Buffer for underline instance data
    pub underline_buffer: gpu::Buffer,
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
            multisample_state: gpu::MultisampleState {
                sample_count: MSAA_SAMPLE_COUNT,
                ..Default::default()
            },
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
            multisample_state: gpu::MultisampleState {
                sample_count: MSAA_SAMPLE_COUNT,
                ..Default::default()
            },
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
            multisample_state: gpu::MultisampleState {
                sample_count: MSAA_SAMPLE_COUNT,
                ..Default::default()
            },
        });

        // Create polychrome sprite instance buffer
        let poly_sprite_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "poly_sprites",
            size: (mem::size_of::<PolychromeSprite>() * MAX_SPRITES_PER_BATCH) as u64,
            memory: gpu::Memory::Shared,
        });

        // Create shadow render pipeline
        let shadow_layout = <ShaderShadowsData as gpu::ShaderData>::layout();
        let shadow_pipeline = gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "shadows",
            data_layouts: &[&shadow_layout],
            vertex: shader.at("vs_shadow"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_shadow")),
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
            multisample_state: gpu::MultisampleState {
                sample_count: MSAA_SAMPLE_COUNT,
                ..Default::default()
            },
        });

        // Create shadow instance buffer
        let shadow_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "shadows",
            size: (mem::size_of::<Shadow>() * MAX_SHADOWS_PER_BATCH) as u64,
            memory: gpu::Memory::Shared,
        });

        // Create path render pipeline
        let path_layout = <ShaderPathsData as gpu::ShaderData>::layout();
        let path_pipeline = gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "paths",
            data_layouts: &[&path_layout],
            vertex: shader.at("vs_path"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_path")),
            primitive: gpu::PrimitiveState {
                topology: gpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            color_targets: &[gpu::ColorTargetState {
                format: surface.info().format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::ALL,
            }],
            multisample_state: gpu::MultisampleState {
                sample_count: MSAA_SAMPLE_COUNT,
                ..Default::default()
            },
        });

        // Create path vertex buffer
        let path_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "path_vertices",
            size: (mem::size_of::<GpuPathVertex>() * MAX_PATH_VERTICES_PER_BATCH) as u64,
            memory: gpu::Memory::Shared,
        });

        // Create underline render pipeline (straight)
        let underline_layout = <ShaderUnderlinesData as gpu::ShaderData>::layout();
        let underline_pipeline = gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "underlines",
            data_layouts: &[&underline_layout],
            vertex: shader.at("vs_underline"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_underline")),
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
            multisample_state: gpu::MultisampleState {
                sample_count: MSAA_SAMPLE_COUNT,
                ..Default::default()
            },
        });

        // Create underline render pipeline (wavy)
        let underline_wavy_pipeline = gpu.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "underlines_wavy",
            data_layouts: &[&underline_layout],
            vertex: shader.at("vs_underline_wavy"),
            vertex_fetches: &[],
            fragment: Some(shader.at("fs_underline_wavy")),
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
            multisample_state: gpu::MultisampleState {
                sample_count: MSAA_SAMPLE_COUNT,
                ..Default::default()
            },
        });

        // Create underline instance buffer
        let underline_buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "underlines",
            size: (mem::size_of::<Underline>() * MAX_UNDERLINES_PER_BATCH) as u64,
            memory: gpu::Memory::Shared,
        });

        // Create texture atlas for sprites and glyphs (Arc for sharing with window)
        let atlas = Arc::new(WebGpuAtlas::new(&gpu));

        // Create MSAA render target for antialiasing
        let msaa_texture = gpu.create_texture(gpu::TextureDesc {
            name: "msaa_target",
            format: surface.info().format,
            size: config.size,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::TARGET,
            external: None,
        });
        let msaa_view = gpu.create_texture_view(
            msaa_texture,
            gpu::TextureViewDesc {
                name: "msaa_view",
                format: surface.info().format,
                dimension: gpu::ViewDimension::D2,
                subresources: &gpu::TextureSubresources::default(),
            },
        );

        *self.0.borrow_mut() = Some(WebRendererState {
            gpu,
            surface,
            surface_config,
            command_encoder,
            last_sync_point: None,
            drawable_size,
            msaa_texture,
            msaa_view,
            globals,
            globals_buffer,
            quad_pipeline,
            quad_buffer,
            mono_sprite_pipeline,
            mono_sprite_buffer,
            poly_sprite_pipeline,
            poly_sprite_buffer,
            shadow_pipeline,
            shadow_buffer,
            path_pipeline,
            path_buffer,
            underline_pipeline,
            underline_wavy_pipeline,
            underline_buffer,
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
            let new_extent = gpu::Extent {
                width: size.width.0 as u32,
                height: size.height.0 as u32,
                depth: 1,
            };
            state.surface_config.size = new_extent;
            state.gpu.reconfigure_surface(&mut state.surface, state.surface_config.clone());

            // Recreate MSAA texture with new size
            state.gpu.destroy_texture(state.msaa_texture);
            state.msaa_texture = state.gpu.create_texture(gpu::TextureDesc {
                name: "msaa_target",
                format: state.surface.info().format,
                size: new_extent,
                array_layer_count: 1,
                mip_level_count: 1,
                sample_count: MSAA_SAMPLE_COUNT,
                dimension: gpu::TextureDimension::D2,
                usage: gpu::TextureUsage::TARGET,
                external: None,
            });
            state.msaa_view = state.gpu.create_texture_view(
                state.msaa_texture,
                gpu::TextureViewDesc {
                    name: "msaa_view",
                    format: state.surface.info().format,
                    dimension: gpu::ViewDimension::D2,
                    subresources: &gpu::TextureSubresources::default(),
                },
            );

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
        measure("      atlas_flush", || state.atlas.flush_uploads());

        // Acquire frame
        let frame = state.surface.acquire_frame();
        if !frame.is_valid() {
            log::warn!("Failed to acquire frame");
            return;
        }

        // Begin encoding
        state.command_encoder.start();

        // Get the texture view for rendering (resolve target)
        let resolve_target = frame.texture_view();

        // Main render pass with MSAA - render to MSAA target, resolve to swapchain
        {
            let mut pass = state.command_encoder.render("main", gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: state.msaa_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                    finish_op: gpu::FinishOp::ResolveTo(resolve_target),
                }],
                depth_stencil: None,
            });

            // Process batches from scene
            // Track buffer offsets for each batch to avoid overwrites
            // (deferred sync means all batches must use different buffer regions)
            let mut quad_buffer_offset: u64 = 0;
            let mut mono_sprite_buffer_offset: u64 = 0;
            let mut poly_sprite_buffer_offset: u64 = 0;
            let mut shadow_buffer_offset: u64 = 0;
            let mut path_buffer_offset: u64 = 0;
            let mut underline_buffer_offset: u64 = 0;

            for batch in scene.batches() {
                match batch {
                    PrimitiveBatch::Quads(quads) => {
                        let new_offset = Self::draw_quads_internal(
                            &mut pass,
                            quads,
                            quad_buffer_offset,
                            &state.globals,
                            state.quad_buffer,
                            &state.quad_pipeline,
                            &state.gpu,
                        );
                        quad_buffer_offset = new_offset;
                    }
                    PrimitiveBatch::Shadows(shadows) => {
                        let new_offset = Self::draw_shadows_internal(
                            &mut pass,
                            shadows,
                            shadow_buffer_offset,
                            &state.globals,
                            state.shadow_buffer,
                            &state.shadow_pipeline,
                            &state.gpu,
                        );
                        shadow_buffer_offset = new_offset;
                    }
                    PrimitiveBatch::MonochromeSprites { texture_id, sprites } => {
                        if let Some(tex_info) = state.atlas.get_texture_info(texture_id) {
                            let new_offset = Self::draw_mono_sprites_internal(
                                &mut pass,
                                sprites,
                                mono_sprite_buffer_offset,
                                &state.globals,
                                tex_info.view,
                                state.atlas_sampler,
                                state.mono_sprite_buffer,
                                &state.mono_sprite_pipeline,
                                &state.gpu,
                            );
                            mono_sprite_buffer_offset = new_offset;
                        } else {
                            log::warn!("No texture info for monochrome sprite batch texture {:?}", texture_id);
                        }
                    }
                    PrimitiveBatch::PolychromeSprites { texture_id, sprites } => {
                        if let Some(tex_info) = state.atlas.get_texture_info(texture_id) {
                            let new_offset = Self::draw_poly_sprites_internal(
                                &mut pass,
                                sprites,
                                poly_sprite_buffer_offset,
                                &state.globals,
                                tex_info.view,
                                state.atlas_sampler,
                                state.poly_sprite_buffer,
                                &state.poly_sprite_pipeline,
                                &state.gpu,
                            );
                            poly_sprite_buffer_offset = new_offset;
                        } else {
                            log::warn!("No texture info for polychrome sprite batch texture {:?}", texture_id);
                        }
                    }
                    PrimitiveBatch::Paths(paths) => {
                        let new_offset = Self::draw_paths_internal(
                            &mut pass,
                            paths,
                            path_buffer_offset,
                            &state.globals,
                            state.path_buffer,
                            &state.path_pipeline,
                            &state.gpu,
                        );
                        path_buffer_offset = new_offset;
                    }
                    PrimitiveBatch::Underlines(underlines) => {
                        let new_offset = Self::draw_underlines_internal(
                            &mut pass,
                            underlines,
                            underline_buffer_offset,
                            &state.globals,
                            state.underline_buffer,
                            &state.underline_pipeline,
                            &state.underline_wavy_pipeline,
                            &state.gpu,
                        );
                        underline_buffer_offset = new_offset;
                    }
                    // TODO: Surfaces primitive type
                    _ => {}
                }
            }
        }

        // Queue frame for presentation
        state.command_encoder.present(frame);

        // Submit
        let sync_point = measure("      gpu_submit", || state.gpu.submit(&mut state.command_encoder));
        state.last_sync_point = Some(sync_point);
    }

    /// WebGPU requires storage buffer offsets to be aligned to minStorageBufferOffsetAlignment (256 bytes)
    const STORAGE_BUFFER_ALIGNMENT: u64 = 256;

    /// Internal helper to draw quads during a render pass
    /// Returns the new buffer offset for the next batch
    #[cfg(target_arch = "wasm32")]
    fn draw_quads_internal(
        pass: &mut gpu::RenderCommandEncoder,
        quads: &[Quad],
        buffer_offset: u64,
        globals: &GlobalParams,
        quad_buffer: gpu::Buffer,
        pipeline: &gpu::RenderPipeline,
        gpu: &gpu::Context,
    ) -> u64 {
        if quads.is_empty() {
            return buffer_offset;
        }

        let count = quads.len().min(MAX_QUADS_PER_BATCH);
        let quad_size = mem::size_of::<Quad>() as u64;
        let data_size = count as u64 * quad_size;

        // Check if we have room in the buffer
        let max_offset = (MAX_QUADS_PER_BATCH as u64) * quad_size;
        if buffer_offset + data_size > max_offset {
            log::warn!("Quad buffer overflow! offset={}, size={}, max={}",
                buffer_offset, data_size, max_offset);
            return buffer_offset;
        }

        log::debug!(
            "draw_quads_internal: drawing {} quads at offset {}, viewport={:?}",
            count, buffer_offset, globals.viewport_size
        );

        // Upload quad data to buffer at the specified offset
        unsafe {
            let dst = (quad_buffer.data() as *mut u8).add(buffer_offset as usize) as *mut Quad;
            ptr::copy_nonoverlapping(quads.as_ptr(), dst, count);
        }
        // Mark the specific range as dirty for efficient sync
        gpu.sync_buffer_range(quad_buffer, buffer_offset, data_size);

        // Bind pipeline and data with the correct buffer offset
        let mut encoder = pass.with(pipeline);
        encoder.bind(
            0,
            &ShaderQuadsData {
                globals: *globals,
                b_quads: gpu::BufferPiece {
                    buffer: quad_buffer,
                    offset: buffer_offset,
                },
            },
        );

        // Draw instanced quads (4 vertices per quad, N instances)
        encoder.draw(0, 4, 0, count as u32);

        // Return the new offset for the next batch, aligned to storage buffer alignment
        let next_offset = buffer_offset + data_size;
        (next_offset + Self::STORAGE_BUFFER_ALIGNMENT - 1) & !(Self::STORAGE_BUFFER_ALIGNMENT - 1)
    }

    /// Internal helper to draw shadows during a render pass
    /// Returns the new buffer offset for the next batch
    #[cfg(target_arch = "wasm32")]
    fn draw_shadows_internal(
        pass: &mut gpu::RenderCommandEncoder,
        shadows: &[Shadow],
        buffer_offset: u64,
        globals: &GlobalParams,
        shadow_buffer: gpu::Buffer,
        pipeline: &gpu::RenderPipeline,
        gpu: &gpu::Context,
    ) -> u64 {
        if shadows.is_empty() {
            return buffer_offset;
        }

        let count = shadows.len().min(MAX_SHADOWS_PER_BATCH);
        let shadow_size = mem::size_of::<Shadow>() as u64;
        let data_size = count as u64 * shadow_size;

        // Check if we have room in the buffer
        let max_offset = (MAX_SHADOWS_PER_BATCH as u64) * shadow_size;
        if buffer_offset + data_size > max_offset {
            log::warn!("Shadow buffer overflow! offset={}, size={}, max={}",
                buffer_offset, data_size, max_offset);
            return buffer_offset;
        }

        log::debug!(
            "draw_shadows_internal: drawing {} shadows at offset {}, viewport={:?}",
            count, buffer_offset, globals.viewport_size
        );

        // Upload shadow data to buffer at the specified offset
        unsafe {
            let dst = (shadow_buffer.data() as *mut u8).add(buffer_offset as usize) as *mut Shadow;
            ptr::copy_nonoverlapping(shadows.as_ptr(), dst, count);
        }
        // Mark the specific range as dirty for efficient sync
        gpu.sync_buffer_range(shadow_buffer, buffer_offset, data_size);

        // Bind pipeline and data with the correct buffer offset
        let mut encoder = pass.with(pipeline);
        encoder.bind(
            0,
            &ShaderShadowsData {
                globals: *globals,
                b_shadows: gpu::BufferPiece {
                    buffer: shadow_buffer,
                    offset: buffer_offset,
                },
            },
        );

        // Draw instanced shadows (4 vertices per shadow, N instances)
        encoder.draw(0, 4, 0, count as u32);

        // Return the new offset for the next batch, aligned to storage buffer alignment
        let next_offset = buffer_offset + data_size;
        (next_offset + Self::STORAGE_BUFFER_ALIGNMENT - 1) & !(Self::STORAGE_BUFFER_ALIGNMENT - 1)
    }

    /// Internal helper to draw paths during a render pass
    /// Returns the new buffer offset for the next batch
    #[cfg(target_arch = "wasm32")]
    fn draw_paths_internal(
        pass: &mut gpu::RenderCommandEncoder,
        paths: &[Path<crate::ScaledPixels>],
        buffer_offset: u64,
        globals: &GlobalParams,
        path_buffer: gpu::Buffer,
        pipeline: &gpu::RenderPipeline,
        gpu: &gpu::Context,
    ) -> u64 {
        if paths.is_empty() {
            return buffer_offset;
        }

        // Count total vertices across all paths
        let total_vertices: usize = paths.iter().map(|p| p.vertices.len()).sum();
        if total_vertices == 0 {
            return buffer_offset;
        }

        let vertex_size = mem::size_of::<GpuPathVertex>() as u64;
        let count = total_vertices.min(MAX_PATH_VERTICES_PER_BATCH);
        let data_size = count as u64 * vertex_size;

        // Check if we have room in the buffer
        let max_offset = (MAX_PATH_VERTICES_PER_BATCH as u64) * vertex_size;
        if buffer_offset + data_size > max_offset {
            log::warn!("Path buffer overflow! offset={}, size={}, max={}",
                buffer_offset, data_size, max_offset);
            return buffer_offset;
        }

        log::debug!(
            "draw_paths_internal: drawing {} paths ({} vertices) at offset {}",
            paths.len(), count, buffer_offset
        );

        // Flatten path vertices into GPU buffer
        unsafe {
            let dst = (path_buffer.data() as *mut u8).add(buffer_offset as usize) as *mut GpuPathVertex;
            let mut vertex_index = 0;

            for path in paths {
                // Get the full Background for gradient support
                let background = &path.color;
                let content_mask = &path.content_mask;
                let bounds = &path.bounds;

                for vertex in &path.vertices {
                    if vertex_index >= count {
                        break;
                    }

                    let gpu_vertex = GpuPathVertex {
                        xy_position_x: vertex.xy_position.x.0,
                        xy_position_y: vertex.xy_position.y.0,
                        st_position_x: vertex.st_position.x,
                        st_position_y: vertex.st_position.y,
                        // content_mask
                        content_mask_origin_x: content_mask.bounds.origin.x.0,
                        content_mask_origin_y: content_mask.bounds.origin.y.0,
                        content_mask_size_width: content_mask.bounds.size.width.0,
                        content_mask_size_height: content_mask.bounds.size.height.0,
                        // path bounds for gradient calculation
                        bounds_origin_x: bounds.origin.x.0,
                        bounds_origin_y: bounds.origin.y.0,
                        bounds_size_width: bounds.size.width.0,
                        bounds_size_height: bounds.size.height.0,
                        // background tag and color space
                        background_tag: background.tag as u32,
                        background_color_space: background.color_space as u32,
                        // solid color
                        solid_h: background.solid.h,
                        solid_s: background.solid.s,
                        solid_l: background.solid.l,
                        solid_a: background.solid.a,
                        // gradient angle
                        gradient_angle: background.gradient_angle_or_pattern_height,
                        // color stops
                        stop0_h: background.colors[0].color.h,
                        stop0_s: background.colors[0].color.s,
                        stop0_l: background.colors[0].color.l,
                        stop0_a: background.colors[0].color.a,
                        stop0_percentage: background.colors[0].percentage,
                        stop1_h: background.colors[1].color.h,
                        stop1_s: background.colors[1].color.s,
                        stop1_l: background.colors[1].color.l,
                        stop1_a: background.colors[1].color.a,
                        stop1_percentage: background.colors[1].percentage,
                        _pad: 0,
                    };

                    ptr::write(dst.add(vertex_index), gpu_vertex);
                    vertex_index += 1;
                }
            }
        }

        // Mark the specific range as dirty for efficient sync
        gpu.sync_buffer_range(path_buffer, buffer_offset, data_size);

        // Bind pipeline and data
        let mut encoder = pass.with(pipeline);
        encoder.bind(
            0,
            &ShaderPathsData {
                globals: *globals,
                b_path_vertices: gpu::BufferPiece {
                    buffer: path_buffer,
                    offset: buffer_offset,
                },
            },
        );

        // Draw triangles (3 vertices per triangle, no instancing)
        encoder.draw(0, count as u32, 0, 1);

        // Return the new offset for the next batch, aligned to storage buffer alignment
        let next_offset = buffer_offset + data_size;
        (next_offset + Self::STORAGE_BUFFER_ALIGNMENT - 1) & !(Self::STORAGE_BUFFER_ALIGNMENT - 1)
    }

    /// Internal helper to draw monochrome sprites during a render pass
    /// Returns the new buffer offset for the next batch
    #[cfg(target_arch = "wasm32")]
    fn draw_mono_sprites_internal(
        pass: &mut gpu::RenderCommandEncoder,
        sprites: &[MonochromeSprite],
        buffer_offset: u64,
        globals: &GlobalParams,
        texture_view: gpu::TextureView,
        sampler: gpu::Sampler,
        sprite_buffer: gpu::Buffer,
        pipeline: &gpu::RenderPipeline,
        gpu: &Rc<gpu::Context>,
    ) -> u64 {
        if sprites.is_empty() {
            return buffer_offset;
        }

        let count = sprites.len().min(MAX_SPRITES_PER_BATCH);
        let sprite_size = mem::size_of::<MonochromeSprite>() as u64;
        let data_size = count as u64 * sprite_size;

        // Check if we have room in the buffer
        let max_offset = (MAX_SPRITES_PER_BATCH as u64) * sprite_size;
        if buffer_offset + data_size > max_offset {
            log::warn!("Mono sprite buffer overflow! offset={}, size={}, max={}",
                buffer_offset, data_size, max_offset);
            return buffer_offset;
        }

        // Upload sprite data to buffer at the specified offset
        unsafe {
            let dst = (sprite_buffer.data() as *mut u8).add(buffer_offset as usize) as *mut MonochromeSprite;
            ptr::copy_nonoverlapping(sprites.as_ptr(), dst, count);
        }
        gpu.sync_buffer_range(sprite_buffer, buffer_offset, data_size);

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
                    offset: buffer_offset,
                },
            },
        );

        // Draw instanced sprites (4 vertices per sprite, N instances)
        encoder.draw(0, 4, 0, count as u32);

        // Return the new offset for the next batch, aligned to storage buffer alignment
        let next_offset = buffer_offset + data_size;
        (next_offset + Self::STORAGE_BUFFER_ALIGNMENT - 1) & !(Self::STORAGE_BUFFER_ALIGNMENT - 1)
    }

    /// Internal helper to draw polychrome sprites during a render pass
    /// Returns the new buffer offset for the next batch
    #[cfg(target_arch = "wasm32")]
    fn draw_poly_sprites_internal(
        pass: &mut gpu::RenderCommandEncoder,
        sprites: &[PolychromeSprite],
        buffer_offset: u64,
        globals: &GlobalParams,
        texture_view: gpu::TextureView,
        sampler: gpu::Sampler,
        sprite_buffer: gpu::Buffer,
        pipeline: &gpu::RenderPipeline,
        gpu: &Rc<gpu::Context>,
    ) -> u64 {
        if sprites.is_empty() {
            return buffer_offset;
        }

        let count = sprites.len().min(MAX_SPRITES_PER_BATCH);
        let sprite_size = mem::size_of::<PolychromeSprite>() as u64;
        let data_size = count as u64 * sprite_size;

        // Check if we have room in the buffer
        let max_offset = (MAX_SPRITES_PER_BATCH as u64) * sprite_size;
        if buffer_offset + data_size > max_offset {
            log::warn!("Poly sprite buffer overflow! offset={}, size={}, max={}",
                buffer_offset, data_size, max_offset);
            return buffer_offset;
        }

        // Upload sprite data to buffer at the specified offset
        unsafe {
            let dst = (sprite_buffer.data() as *mut u8).add(buffer_offset as usize) as *mut PolychromeSprite;
            ptr::copy_nonoverlapping(sprites.as_ptr(), dst, count);
        }
        gpu.sync_buffer_range(sprite_buffer, buffer_offset, data_size);

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
                    offset: buffer_offset,
                },
            },
        );

        // Draw instanced sprites (4 vertices per sprite, N instances)
        encoder.draw(0, 4, 0, count as u32);

        // Return the new offset for the next batch, aligned to storage buffer alignment
        let next_offset = buffer_offset + data_size;
        (next_offset + Self::STORAGE_BUFFER_ALIGNMENT - 1) & !(Self::STORAGE_BUFFER_ALIGNMENT - 1)
    }

    /// Internal helper to draw underlines during a render pass
    /// Returns the new buffer offset for the next batch
    #[cfg(target_arch = "wasm32")]
    fn draw_underlines_internal(
        pass: &mut gpu::RenderCommandEncoder,
        underlines: &[Underline],
        buffer_offset: u64,
        globals: &GlobalParams,
        underline_buffer: gpu::Buffer,
        straight_pipeline: &gpu::RenderPipeline,
        wavy_pipeline: &gpu::RenderPipeline,
        gpu: &gpu::Context,
    ) -> u64 {
        if underlines.is_empty() {
            return buffer_offset;
        }

        // Separate straight and wavy underlines for different pipelines
        let straight: Vec<_> = underlines.iter().filter(|u| u.wavy == 0).cloned().collect();
        let wavy: Vec<_> = underlines.iter().filter(|u| u.wavy != 0).cloned().collect();

        let mut current_offset = buffer_offset;
        let underline_size = mem::size_of::<Underline>() as u64;

        // Draw straight underlines
        if !straight.is_empty() {
            let count = straight.len().min(MAX_UNDERLINES_PER_BATCH);
            let data_size = count as u64 * underline_size;

            // Check if we have room in the buffer
            let max_offset = (MAX_UNDERLINES_PER_BATCH as u64) * underline_size;
            if current_offset + data_size > max_offset {
                log::warn!("Underline buffer overflow! offset={}, size={}, max={}",
                    current_offset, data_size, max_offset);
                return current_offset;
            }

            log::debug!(
                "draw_underlines_internal: drawing {} straight underlines at offset {}",
                count, current_offset
            );

            // Upload underline data to buffer
            unsafe {
                let dst = (underline_buffer.data() as *mut u8).add(current_offset as usize) as *mut Underline;
                ptr::copy_nonoverlapping(straight.as_ptr(), dst, count);
            }
            gpu.sync_buffer_range(underline_buffer, current_offset, data_size);

            // Bind pipeline and data
            let mut encoder = pass.with(straight_pipeline);
            encoder.bind(
                0,
                &ShaderUnderlinesData {
                    globals: *globals,
                    b_underlines: gpu::BufferPiece {
                        buffer: underline_buffer,
                        offset: current_offset,
                    },
                },
            );

            // Draw instanced underlines (4 vertices per underline, N instances)
            encoder.draw(0, 4, 0, count as u32);

            // Update offset for next batch
            let next_offset = current_offset + data_size;
            current_offset = (next_offset + Self::STORAGE_BUFFER_ALIGNMENT - 1) & !(Self::STORAGE_BUFFER_ALIGNMENT - 1);
        }

        // Draw wavy underlines
        if !wavy.is_empty() {
            let count = wavy.len().min(MAX_UNDERLINES_PER_BATCH);
            let data_size = count as u64 * underline_size;

            // Check if we have room in the buffer
            let max_offset = (MAX_UNDERLINES_PER_BATCH as u64) * underline_size;
            if current_offset + data_size > max_offset {
                log::warn!("Underline buffer overflow (wavy)! offset={}, size={}, max={}",
                    current_offset, data_size, max_offset);
                return current_offset;
            }

            log::debug!(
                "draw_underlines_internal: drawing {} wavy underlines at offset {}",
                count, current_offset
            );

            // Upload underline data to buffer
            unsafe {
                let dst = (underline_buffer.data() as *mut u8).add(current_offset as usize) as *mut Underline;
                ptr::copy_nonoverlapping(wavy.as_ptr(), dst, count);
            }
            gpu.sync_buffer_range(underline_buffer, current_offset, data_size);

            // Bind pipeline and data
            let mut encoder = pass.with(wavy_pipeline);
            encoder.bind(
                0,
                &ShaderUnderlinesData {
                    globals: *globals,
                    b_underlines: gpu::BufferPiece {
                        buffer: underline_buffer,
                        offset: current_offset,
                    },
                },
            );

            // Draw instanced underlines (4 vertices per underline, N instances)
            encoder.draw(0, 4, 0, count as u32);

            // Update offset for next batch
            let next_offset = current_offset + data_size;
            current_offset = (next_offset + Self::STORAGE_BUFFER_ALIGNMENT - 1) & !(Self::STORAGE_BUFFER_ALIGNMENT - 1);
        }

        current_offset
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

        // Get the texture view for rendering (resolve target)
        let resolve_target = frame.texture_view();

        // Main render pass with MSAA
        {
            let mut pass = state.command_encoder.render("main", gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: state.msaa_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::ResolveTo(resolve_target),
                }],
                depth_stencil: None,
            });

            Self::draw_quads_internal(
                &mut pass,
                &[quad],
                0, // buffer_offset
                &state.globals,
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

    /// Draw test text at the specified position
    ///
    /// Uses Canvas 2D to render text and displays it as a monochrome sprite.
    /// This tests the full text rendering pipeline: canvas → atlas → sprite.
    #[cfg(target_arch = "wasm32")]
    pub fn draw_test_text(&self, text: &str, x: f32, y: f32, font_size: f32, color: [f32; 4]) {
        use crate::{
            AtlasKey, Bounds, ContentMask, Hsla, ScaledPixels,
            scene::{DrawOrder, TransformationMatrix},
        };
        use wasm_bindgen::JsCast;

        let mut state_ref = self.0.borrow_mut();
        let Some(state) = state_ref.as_mut() else {
            log::warn!("WebRenderer::draw_test_text called before initialization");
            return;
        };

        // Wait for previous frame
        if let Some(ref sp) = state.last_sync_point {
            let _ = state.gpu.wait_for(sp, 1000);
        }

        // Create offscreen canvas for text rendering
        let document = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");
        let canvas = document
            .create_element("canvas")
            .expect("failed to create canvas")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("not a canvas");
        let context = canvas
            .get_context("2d")
            .expect("failed to get 2d context")
            .expect("no 2d context")
            .dyn_into::<web_sys::CanvasRenderingContext2d>()
            .expect("not a 2d context");

        // Measure text to determine canvas size
        let font = format!("{}px system-ui, sans-serif", font_size);
        context.set_font(&font);
        let metrics = context.measure_text(text).expect("measure_text failed");
        let text_width = metrics.width().ceil() as u32 + 4; // Add padding
        let text_height = (font_size * 1.3).ceil() as u32 + 4;

        canvas.set_width(text_width);
        canvas.set_height(text_height);

        // Re-set font after resize (canvas resize clears state)
        context.set_font(&font);
        context.set_fill_style_str("white");
        context.set_text_baseline("top");

        // Clear and draw text
        context.clear_rect(0.0, 0.0, text_width as f64, text_height as f64);
        context.fill_text(text, 2.0, 2.0).expect("fill_text failed");

        // Get image data and convert to grayscale
        let image_data = context
            .get_image_data(0.0, 0.0, text_width as f64, text_height as f64)
            .expect("get_image_data failed");
        let rgba_data = image_data.data();

        // Convert RGBA to grayscale (using alpha channel)
        let mut grayscale = Vec::with_capacity((text_width * text_height) as usize);
        for i in (0..rgba_data.len()).step_by(4) {
            grayscale.push(rgba_data[i + 3]); // Use alpha channel
        }

        // Create atlas key for this text using a glyph key with fake params
        // Use hash of text+size as a unique "glyph id"
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher as _};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        (font_size as u32).hash(&mut hasher);
        let hash = hasher.finish();

        let key = AtlasKey::Glyph(crate::RenderGlyphParams {
            font_id: crate::FontId(0),
            glyph_id: crate::GlyphId(hash as u32),
            font_size: crate::Pixels(font_size),
            subpixel_variant: crate::Point { x: 0, y: 0 },
            scale_factor: 1.0,
            is_emoji: false,
        });

        // Upload to atlas
        let tile = state
            .atlas
            .get_or_insert_with(&key, &mut || {
                Ok(Some((
                    Size {
                        width: DevicePixels(text_width as i32),
                        height: DevicePixels(text_height as i32),
                    },
                    std::borrow::Cow::Owned(grayscale.clone()),
                )))
            })
            .expect("atlas insert failed")
            .expect("no tile");

        // Flush atlas uploads
        state.atlas.flush_uploads();

        // Create MonochromeSprite
        let sprite = MonochromeSprite {
            order: DrawOrder::default(),
            pad: 0,
            bounds: Bounds {
                origin: crate::Point {
                    x: ScaledPixels(x),
                    y: ScaledPixels(y),
                },
                size: crate::Size {
                    width: ScaledPixels(text_width as f32),
                    height: ScaledPixels(text_height as f32),
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
            color: Hsla {
                h: color[0],
                s: color[1],
                l: color[2],
                a: color[3],
            },
            tile,
            transformation: TransformationMatrix::unit(),
        };

        // Acquire frame
        let frame = state.surface.acquire_frame();
        if !frame.is_valid() {
            log::warn!("Failed to acquire frame");
            return;
        }

        // Begin encoding
        state.command_encoder.start();
        let resolve_target = frame.texture_view();

        // Main render pass with MSAA
        {
            let mut pass = state.command_encoder.render("main", gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: state.msaa_view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::ResolveTo(resolve_target),
                }],
                depth_stencil: None,
            });

            // Get texture info for the sprite's texture
            if let Some(tex_info) = state.atlas.get_texture_info(sprite.tile.texture_id) {
                Self::draw_mono_sprites_internal(
                    &mut pass,
                    &[sprite],
                    0, // buffer_offset
                    &state.globals,
                    tex_info.view,
                    state.atlas_sampler,
                    state.mono_sprite_buffer,
                    &state.mono_sprite_pipeline,
                    &state.gpu,
                );
            } else {
                log::warn!("No texture info for sprite tile");
            }
        }

        // Queue frame for presentation
        state.command_encoder.present(frame);

        // Submit
        let sync_point = state.gpu.submit(&mut state.command_encoder);
        state.last_sync_point = Some(sync_point);

        log::info!("Drew text '{}' at ({}, {}) size {}x{}", text, x, y, text_width, text_height);
    }

    /// Draw test text (non-WASM stub)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn draw_test_text(&self, _text: &str, _x: f32, _y: f32, _font_size: f32, _color: [f32; 4]) {
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
