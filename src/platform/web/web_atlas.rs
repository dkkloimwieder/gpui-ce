//! WebGPU-backed texture atlas for GPUI WASM
//!
//! Provides GPU texture atlas management for glyph/sprite rendering in the browser.
//! Uses etagere for rectangle packing and blade-graphics for GPU operations.
//!
//! Note: Uses Rc/RefCell instead of Arc/Mutex since WASM is single-threaded
//! and wgpu types don't implement Send/Sync on WASM.

use crate::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, DevicePixels, PlatformAtlas,
    Point, Size, platform::AtlasTextureList,
};
use anyhow::Result;
use blade_graphics as gpu;
use collections::FxHashMap;
use etagere::BucketedAtlasAllocator;
use std::{borrow::Cow, cell::RefCell, ops, rc::Rc};

/// GPU-backed texture atlas for web platform
pub struct WebGpuAtlas(RefCell<WebGpuAtlasState>);

// SAFETY: WebGpuAtlas is only used on WASM which is single-threaded.
// These impls are required because PlatformAtlas requires Send + Sync,
// but RefCell doesn't implement them. On WASM, these are safe because
// there's only one thread.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for WebGpuAtlas {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WebGpuAtlas {}

/// Pending texture upload
struct PendingUpload {
    id: AtlasTextureId,
    bounds: Bounds<DevicePixels>,
    data: Vec<u8>,
}

/// Cache statistics for debugging
#[derive(Default)]
struct CacheStats {
    hits: u32,
    misses: u32,
}

/// Internal state of the atlas
struct WebGpuAtlasState {
    gpu: Rc<gpu::Context>,
    storage: WebGpuAtlasStorage,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    uploads: Vec<PendingUpload>,
    stats: CacheStats,
}

/// Information about an atlas texture for binding in shaders
pub struct WebAtlasTextureInfo {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
}

impl WebGpuAtlas {
    /// Create a new GPU-backed atlas
    pub fn new(gpu: &Rc<gpu::Context>) -> Self {
        WebGpuAtlas(RefCell::new(WebGpuAtlasState {
            gpu: Rc::clone(gpu),
            storage: WebGpuAtlasStorage::default(),
            tiles_by_key: Default::default(),
            uploads: Vec::new(),
            stats: CacheStats::default(),
        }))
    }

    /// Flush pending texture uploads using the command encoder
    ///
    /// This should be called before rendering to ensure all texture data is uploaded.
    /// Uses queue.write_texture() for simplicity on web.
    pub fn flush_uploads(&self) {
        let mut state = self.0.borrow_mut();
        state.flush_uploads();
    }

    /// Get texture info for binding in shaders
    pub fn get_texture_info(&self, id: AtlasTextureId) -> Option<WebAtlasTextureInfo> {
        let state = self.0.borrow();
        state.storage.get(id).map(|texture| WebAtlasTextureInfo {
            texture: texture.raw,
            view: texture.raw_view,
        })
    }

    /// Get all monochrome textures for binding
    pub fn monochrome_textures(&self) -> Vec<WebAtlasTextureInfo> {
        let state = self.0.borrow();
        state.storage.monochrome_textures.textures
            .iter()
            .filter_map(|t| t.as_ref())
            .map(|t| WebAtlasTextureInfo {
                texture: t.raw,
                view: t.raw_view,
            })
            .collect()
    }

    /// Get all polychrome textures for binding
    pub fn polychrome_textures(&self) -> Vec<WebAtlasTextureInfo> {
        let state = self.0.borrow();
        state.storage.polychrome_textures.textures
            .iter()
            .filter_map(|t| t.as_ref())
            .map(|t| WebAtlasTextureInfo {
                texture: t.raw,
                view: t.raw_view,
            })
            .collect()
    }

    /// Destroy all atlas resources
    pub fn destroy(&self) {
        let mut state = self.0.borrow_mut();
        let gpu = state.gpu.clone();
        state.storage.destroy(&gpu);
    }
}

impl PlatformAtlas for WebGpuAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut state = self.0.borrow_mut();

        // Return cached tile if exists
        if let Some(tile) = state.tiles_by_key.get(key).cloned() {
            state.stats.hits += 1;
            return Ok(Some(tile));
        }

        state.stats.misses += 1;

        // Build the tile data
        let Some((size, bytes)) = build()? else {
            return Ok(None);
        };

        // Allocate space in atlas
        let tile = state.allocate(size, key.texture_kind());

        // Queue upload
        state.uploads.push(PendingUpload {
            id: tile.texture_id,
            bounds: tile.bounds,
            data: bytes.into_owned(),
        });

        // Cache the tile
        state.tiles_by_key.insert(key.clone(), tile.clone());

        Ok(Some(tile))
    }

    fn remove(&self, key: &AtlasKey) {
        let mut state = self.0.borrow_mut();

        let Some(tile) = state.tiles_by_key.remove(key) else {
            return;
        };

        // Decrement reference count on texture
        if let Some(texture) = state.storage.get_mut(tile.texture_id) {
            texture.decrement_ref_count();
            // Note: We don't immediately free textures - they can be reused
        }
    }
}

impl WebGpuAtlasState {
    /// Allocate space for a tile in the atlas
    fn allocate(&mut self, size: Size<DevicePixels>, kind: AtlasTextureKind) -> AtlasTile {
        // Try to allocate in existing textures
        let textures = &mut self.storage[kind];
        if let Some(tile) = textures.iter_mut().rev().find_map(|t| t.allocate(size)) {
            return tile;
        }

        // Create new texture
        let texture = self.push_texture(size, kind);
        texture.allocate(size).expect("Failed to allocate in new texture")
    }

    /// Create a new atlas texture
    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut WebGpuAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };

        let size = min_size.max(&DEFAULT_ATLAS_SIZE);

        let (format, bytes_per_pixel) = match kind {
            AtlasTextureKind::Monochrome => (gpu::TextureFormat::R8Unorm, 1),
            AtlasTextureKind::Polychrome => (gpu::TextureFormat::Bgra8Unorm, 4),
        };

        let raw = self.gpu.create_texture(gpu::TextureDesc {
            name: "web_atlas",
            format,
            size: gpu::Extent {
                width: size.width.into(),
                height: size.height.into(),
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage: gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
            external: None,
        });

        let raw_view = self.gpu.create_texture_view(
            raw,
            gpu::TextureViewDesc {
                name: "web_atlas_view",
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        let texture_list = &mut self.storage[kind];
        let index = texture_list.free_list.pop();

        let atlas_texture = WebGpuAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: BucketedAtlasAllocator::new(size.into()),
            size,
            format,
            bytes_per_pixel,
            raw,
            raw_view,
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix).unwrap().as_mut().unwrap()
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut().unwrap().as_mut().unwrap()
        }
    }

    /// Flush pending uploads to GPU
    fn flush_uploads(&mut self) {
        // Log cache stats for this frame
        let total = self.stats.hits + self.stats.misses;
        if total > 0 {
            let hit_rate = (self.stats.hits as f32 / total as f32) * 100.0;
            log::info!(
                "glyph cache: {} hits, {} misses ({:.1}% hit rate), {} total tiles",
                self.stats.hits,
                self.stats.misses,
                hit_rate,
                self.tiles_by_key.len()
            );
        }
        // Reset stats for next frame
        self.stats = CacheStats::default();

        for upload in self.uploads.drain(..) {
            let Some(texture) = self.storage.get(upload.id) else {
                continue;
            };

            // Use wgpu's write_texture for direct upload (simpler than staging buffer on web)
            let bytes_per_row = upload.bounds.size.width.0 as u32 * texture.bytes_per_pixel as u32;

            self.gpu.write_texture(
                gpu::TexturePiece {
                    texture: texture.raw,
                    mip_level: 0,
                    array_layer: 0,
                    origin: [
                        upload.bounds.origin.x.into(),
                        upload.bounds.origin.y.into(),
                        0,
                    ],
                },
                &upload.data,
                gpu::TextureDataLayout {
                    bytes_per_row,
                    rows_per_image: upload.bounds.size.height.0 as u32,
                },
                gpu::Extent {
                    width: upload.bounds.size.width.into(),
                    height: upload.bounds.size.height.into(),
                    depth: 1,
                },
            );
        }
    }
}

/// Storage for atlas textures by kind
#[derive(Default)]
struct WebGpuAtlasStorage {
    monochrome_textures: AtlasTextureList<WebGpuAtlasTexture>,
    polychrome_textures: AtlasTextureList<WebGpuAtlasTexture>,
}

impl ops::Index<AtlasTextureKind> for WebGpuAtlasStorage {
    type Output = AtlasTextureList<WebGpuAtlasTexture>;
    fn index(&self, kind: AtlasTextureKind) -> &Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }
}

impl ops::IndexMut<AtlasTextureKind> for WebGpuAtlasStorage {
    fn index_mut(&mut self, kind: AtlasTextureKind) -> &mut Self::Output {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }
}

impl WebGpuAtlasStorage {
    fn get(&self, id: AtlasTextureId) -> Option<&WebGpuAtlasTexture> {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };
        textures.textures.get(id.index as usize)?.as_ref()
    }

    fn get_mut(&mut self, id: AtlasTextureId) -> Option<&mut WebGpuAtlasTexture> {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        };
        textures.textures.get_mut(id.index as usize)?.as_mut()
    }

    fn destroy(&mut self, gpu: &gpu::Context) {
        for texture in self.monochrome_textures.textures.drain(..).flatten() {
            gpu.destroy_texture_view(texture.raw_view);
            gpu.destroy_texture(texture.raw);
        }
        for texture in self.polychrome_textures.textures.drain(..).flatten() {
            gpu.destroy_texture_view(texture.raw_view);
            gpu.destroy_texture(texture.raw);
        }
    }
}

/// A single atlas texture with its allocator
struct WebGpuAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    #[allow(dead_code)]
    size: Size<DevicePixels>,
    #[allow(dead_code)]
    format: gpu::TextureFormat,
    bytes_per_pixel: u8,
    raw: gpu::Texture,
    raw_view: gpu::TextureView,
    live_atlas_keys: u32,
}

impl WebGpuAtlasTexture {
    /// Allocate space for a tile
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size.into())?;
        self.live_atlas_keys += 1;

        Some(AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            padding: 0,
            bounds: Bounds {
                origin: allocation.rectangle.min.into(),
                size,
            },
        })
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys = self.live_atlas_keys.saturating_sub(1);
    }
}

// Conversion helpers for etagere types
impl From<Size<DevicePixels>> for etagere::Size {
    fn from(size: Size<DevicePixels>) -> Self {
        etagere::Size::new(size.width.into(), size.height.into())
    }
}

impl From<etagere::Point> for Point<DevicePixels> {
    fn from(value: etagere::Point) -> Self {
        Point {
            x: DevicePixels::from(value.x),
            y: DevicePixels::from(value.y),
        }
    }
}
