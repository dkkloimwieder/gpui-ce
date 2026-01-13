//! Web-based text system using Canvas 2D for text rendering
//!
//! Uses the browser's text rendering capabilities via Canvas 2D API
//! for font metrics, text layout, and glyph rasterization.

use crate::{
    Bounds, DevicePixels, Font, FontId, FontMetrics, FontRun, FontStyle, GlyphId, LineLayout,
    Pixels, PlatformTextSystem, Point, RenderGlyphParams, ShapedGlyph, ShapedRun, Size, point,
    px, size,
};
use anyhow::Result;
use collections::HashMap;
use parking_lot::RwLock;
use std::borrow::Cow;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Font information stored for each registered font
#[derive(Clone, Debug)]
struct FontInfo {
    /// CSS font family name
    family: String,
    /// Font weight (100-900)
    weight: f32,
    /// Font style (normal/italic/oblique)
    style: FontStyle,
    /// Cached CSS font string (e.g., "italic 700 16px Arial")
    css_template: String,
}

impl FontInfo {
    fn new(family: String, weight: f32, style: FontStyle) -> Self {
        let style_str = match style {
            FontStyle::Normal => "",
            FontStyle::Italic => "italic ",
            FontStyle::Oblique => "oblique ",
        };
        // Template with placeholder for size
        let css_template = format!("{style_str}{weight} {{size}}px {family}");
        Self {
            family,
            weight,
            style,
            css_template,
        }
    }

    /// Get CSS font string for a specific size
    fn css_font(&self, size: f32) -> String {
        self.css_template.replace("{size}", &size.to_string())
    }
}

/// Web-based text system state
struct WebTextSystemState {
    /// Registered fonts indexed by FontId
    fonts: Vec<FontInfo>,
    /// Map from (family, weight, style) to FontId
    font_cache: HashMap<(String, u32, FontStyle), FontId>,
    /// Offscreen canvas for rasterization
    #[cfg(target_arch = "wasm32")]
    canvas: web_sys::HtmlCanvasElement,
    #[cfg(target_arch = "wasm32")]
    context: web_sys::CanvasRenderingContext2d,
    /// Default font ID (system UI font)
    default_font_id: FontId,
}

/// Web text system using Canvas 2D API
pub struct WebTextSystem(RwLock<WebTextSystemState>);

impl WebTextSystem {
    /// Create a new web text system
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        let (canvas, context) = {
            let document = web_sys::window()
                .expect("no window")
                .document()
                .expect("no document");
            let canvas = document
                .create_element("canvas")
                .expect("failed to create canvas")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("not a canvas");
            // Start with reasonable size, will resize as needed
            canvas.set_width(512);
            canvas.set_height(128);
            let context = canvas
                .get_context("2d")
                .expect("failed to get 2d context")
                .expect("no 2d context")
                .dyn_into::<web_sys::CanvasRenderingContext2d>()
                .expect("not a 2d context");
            (canvas, context)
        };

        // Register default system font
        let default_font = FontInfo::new(
            "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
                .to_string(),
            400.0,
            FontStyle::Normal,
        );

        let state = WebTextSystemState {
            fonts: vec![default_font],
            font_cache: HashMap::default(),
            #[cfg(target_arch = "wasm32")]
            canvas,
            #[cfg(target_arch = "wasm32")]
            context,
            default_font_id: FontId(0),
        };

        Self(RwLock::new(state))
    }

    /// Map system font name to web-compatible font family
    fn web_font_family(family: &str) -> String {
        if family == ".SystemUIFont" || family.is_empty() {
            "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
                .to_string()
        } else {
            // Quote family name if it contains spaces
            if family.contains(' ') {
                format!("'{family}'")
            } else {
                family.to_string()
            }
        }
    }
}

impl Default for WebTextSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformTextSystem for WebTextSystem {
    fn add_fonts(&self, _fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        // Web fonts are loaded via CSS @font-face or font loading API
        // For now, we rely on CSS-loaded fonts
        Ok(())
    }

    fn all_font_names(&self) -> Vec<String> {
        // Return commonly available web fonts
        vec![
            "system-ui".to_string(),
            "sans-serif".to_string(),
            "serif".to_string(),
            "monospace".to_string(),
            "Arial".to_string(),
            "Helvetica".to_string(),
            "Times New Roman".to_string(),
            "Courier New".to_string(),
            "Georgia".to_string(),
            "Verdana".to_string(),
        ]
    }

    fn font_id(&self, font: &Font) -> Result<FontId> {
        let mut state = self.0.write();

        let family = Self::web_font_family(&font.family);
        let weight = (font.weight.0 as u32 / 100) * 100; // Round to nearest 100
        let style = font.style;

        // Check cache first
        let cache_key = (family.clone(), weight, style);
        if let Some(&id) = state.font_cache.get(&cache_key) {
            return Ok(id);
        }

        // Create new font entry
        let font_info = FontInfo::new(family.clone(), weight as f32, style);
        let id = FontId(state.fonts.len());
        state.fonts.push(font_info);
        state.font_cache.insert(cache_key, id);

        Ok(id)
    }

    fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        #[cfg(target_arch = "wasm32")]
        {
            let state = self.0.read();
            let font_info = &state.fonts[font_id.0];

            // Use a reference size for metrics
            let ref_size = 1000.0;
            state.context.set_font(&font_info.css_font(ref_size));

            // Measure 'M' for em-based metrics
            if let Ok(metrics) = state.context.measure_text("M") {
                let actual_bounding_box_ascent = metrics.actual_bounding_box_ascent();
                let actual_bounding_box_descent = metrics.actual_bounding_box_descent();
                let font_bounding_box_ascent = metrics.font_bounding_box_ascent();
                let font_bounding_box_descent = metrics.font_bounding_box_descent();

                // Use font bounding box if available, otherwise actual
                let ascent = if font_bounding_box_ascent > 0.0 {
                    font_bounding_box_ascent
                } else {
                    actual_bounding_box_ascent
                };
                let descent = if font_bounding_box_descent > 0.0 {
                    font_bounding_box_descent
                } else {
                    actual_bounding_box_descent
                };

                return FontMetrics {
                    units_per_em: ref_size as u32,
                    ascent: ascent as f32,
                    descent: -(descent as f32), // GPUI uses negative descent
                    line_gap: 0.0,
                    underline_position: -(descent as f32 * 0.5),
                    underline_thickness: ref_size as f32 * 0.05,
                    cap_height: ascent as f32 * 0.7,
                    x_height: ascent as f32 * 0.5,
                    bounding_box: Bounds {
                        origin: point(0.0, -(descent as f32)),
                        size: size(metrics.width() as f32, (ascent + descent) as f32),
                    },
                };
            }
        }

        // Fallback metrics (same as NoopTextSystem)
        FontMetrics {
            units_per_em: 1000,
            ascent: 800.0,
            descent: -200.0,
            line_gap: 0.0,
            underline_position: -100.0,
            underline_thickness: 50.0,
            cap_height: 700.0,
            x_height: 500.0,
            bounding_box: Bounds {
                origin: point(0.0, -200.0),
                size: size(1000.0, 1000.0),
            },
        }
    }

    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        #[cfg(target_arch = "wasm32")]
        {
            let state = self.0.read();
            let font_info = &state.fonts[font_id.0];

            // Use reference size
            let ref_size = 1000.0;
            state.context.set_font(&font_info.css_font(ref_size));

            // Convert glyph_id back to char
            if let Some(ch) = char::from_u32(glyph_id.0) {
                let s = ch.to_string();
                if let Ok(metrics) = state.context.measure_text(&s) {
                    return Ok(Bounds {
                        origin: point(0.0, 0.0),
                        size: size(
                            metrics.width() as f32,
                            (metrics.actual_bounding_box_ascent()
                                + metrics.actual_bounding_box_descent())
                                as f32,
                        ),
                    });
                }
            }
        }

        Ok(Bounds {
            origin: point(0.0, 0.0),
            size: size(500.0, 1000.0),
        })
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        #[cfg(target_arch = "wasm32")]
        {
            let state = self.0.read();
            let font_info = &state.fonts[font_id.0];

            // Use reference size
            let ref_size = 1000.0;
            state.context.set_font(&font_info.css_font(ref_size));

            // Convert glyph_id back to char
            if let Some(ch) = char::from_u32(glyph_id.0) {
                let s = ch.to_string();
                if let Ok(metrics) = state.context.measure_text(&s) {
                    return Ok(size(metrics.width() as f32, 0.0));
                }
            }
        }

        Ok(size(600.0, 0.0))
    }

    fn glyph_for_char(&self, _font_id: FontId, ch: char) -> Option<GlyphId> {
        // For web, we use the Unicode codepoint as the glyph ID
        // This simplifies the mapping since Canvas 2D works with strings
        Some(GlyphId(ch as u32))
    }

    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        #[cfg(target_arch = "wasm32")]
        {
            let state = self.0.read();
            let font_info = &state.fonts[params.font_id.0];

            let scaled_size = params.font_size.0 * params.scale_factor;
            state.context.set_font(&font_info.css_font(scaled_size));

            if let Some(ch) = char::from_u32(params.glyph_id.0) {
                let s = ch.to_string();
                if let Ok(metrics) = state.context.measure_text(&s) {
                    let width = metrics.width().ceil() as i32;
                    let ascent = metrics.actual_bounding_box_ascent().ceil() as i32;
                    let descent = metrics.actual_bounding_box_descent().ceil() as i32;

                    // Add padding for antialiasing
                    let padding = 2;
                    return Ok(Bounds {
                        origin: Point {
                            x: DevicePixels(-padding),
                            y: DevicePixels(-(ascent + padding)),
                        },
                        size: Size {
                            width: DevicePixels(width + padding * 2),
                            height: DevicePixels(ascent + descent + padding * 2),
                        },
                    });
                }
            }
        }

        Ok(Bounds::default())
    }

    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        let width = raster_bounds.size.width.0 as u32;
        let height = raster_bounds.size.height.0 as u32;

        if width == 0 || height == 0 {
            return Ok((raster_bounds.size, Vec::new()));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let state = self.0.read();
            let font_info = &state.fonts[params.font_id.0];

            // Ensure canvas is large enough
            if state.canvas.width() < width || state.canvas.height() < height {
                state.canvas.set_width(width.max(512));
                state.canvas.set_height(height.max(128));
            }

            // Clear canvas
            state
                .context
                .clear_rect(0.0, 0.0, width as f64, height as f64);

            // Set up font and draw
            let scaled_size = params.font_size.0 * params.scale_factor;
            state.context.set_font(&font_info.css_font(scaled_size));
            state.context.set_fill_style_str("white");
            state.context.set_text_baseline("alphabetic");

            if let Some(ch) = char::from_u32(params.glyph_id.0) {
                let s = ch.to_string();

                // Draw at position that accounts for bounds origin
                let x = -raster_bounds.origin.x.0 as f64;
                let y = -raster_bounds.origin.y.0 as f64;

                state
                    .context
                    .fill_text(&s, x, y)
                    .map_err(|e| anyhow::anyhow!("fill_text failed: {:?}", e))?;

                // Get image data
                let image_data = state
                    .context
                    .get_image_data(0.0, 0.0, width as f64, height as f64)
                    .map_err(|e| anyhow::anyhow!("get_image_data failed: {:?}", e))?;

                let rgba_data = image_data.data();

                // Convert RGBA to grayscale (alpha channel for monochrome glyphs)
                // For text, we use the alpha channel since we draw white text
                let mut grayscale = Vec::with_capacity((width * height) as usize);
                for i in (0..rgba_data.len()).step_by(4) {
                    // Use alpha channel for coverage
                    grayscale.push(rgba_data[i + 3]);
                }

                return Ok((raster_bounds.size, grayscale));
            }
        }

        // Return empty bitmap for non-WASM or failed rasterization
        Ok((raster_bounds.size, vec![0u8; (width * height) as usize]))
    }

    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
        if text.is_empty() {
            return LineLayout {
                font_size,
                width: px(0.),
                ascent: px(0.),
                descent: px(0.),
                runs: Vec::new(),
                len: 0,
            };
        }

        #[cfg(target_arch = "wasm32")]
        {
            let state = self.0.read();

            let mut shaped_runs = Vec::new();
            let mut total_width = 0.0f32;
            let mut max_ascent = 0.0f32;
            let mut max_descent = 0.0f32;

            let mut char_offset = 0usize;
            let mut position_x = 0.0f32;

            for run in runs {
                let font_info = &state.fonts[run.font_id.0];
                state.context.set_font(&font_info.css_font(font_size.0));

                // Get byte range for this run
                let run_end = (char_offset + run.len).min(text.len());
                let run_text = &text[char_offset..run_end];

                // Measure run metrics
                if let Ok(metrics) = state.context.measure_text(run_text) {
                    let ascent = metrics.font_bounding_box_ascent();
                    let descent = metrics.font_bounding_box_descent();

                    if ascent > 0.0 {
                        max_ascent = max_ascent.max(ascent as f32);
                    } else {
                        max_ascent =
                            max_ascent.max(metrics.actual_bounding_box_ascent() as f32);
                    }
                    if descent > 0.0 {
                        max_descent = max_descent.max(descent as f32);
                    } else {
                        max_descent =
                            max_descent.max(metrics.actual_bounding_box_descent() as f32);
                    }
                }

                // Shape each character in the run
                let mut glyphs = Vec::new();
                for (byte_idx, ch) in run_text.char_indices() {
                    let char_str = ch.to_string();

                    // Detect emoji (simple heuristic)
                    let is_emoji = ch as u32 > 0x1F000
                        || (ch as u32 >= 0x2600 && ch as u32 <= 0x27BF)
                        || (ch as u32 >= 0xFE00 && ch as u32 <= 0xFE0F);

                    glyphs.push(ShapedGlyph {
                        id: GlyphId(ch as u32),
                        position: point(px(position_x), px(0.)),
                        index: char_offset + byte_idx,
                        is_emoji,
                    });

                    // Measure character advance
                    if let Ok(metrics) = state.context.measure_text(&char_str) {
                        position_x += metrics.width() as f32;
                    }
                }

                if !glyphs.is_empty() {
                    shaped_runs.push(ShapedRun {
                        font_id: run.font_id,
                        glyphs,
                    });
                }

                char_offset = run_end;
            }

            total_width = position_x;

            // If no font metrics were obtained, use reasonable defaults
            if max_ascent == 0.0 {
                max_ascent = font_size.0 * 0.8;
            }
            if max_descent == 0.0 {
                max_descent = font_size.0 * 0.2;
            }

            return LineLayout {
                font_size,
                width: px(total_width),
                ascent: px(max_ascent),
                descent: px(max_descent),
                runs: shaped_runs,
                len: text.len(),
            };
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // Non-WASM fallback (shouldn't be used in practice)
            let metrics = self.font_metrics(FontId(0));
            let em_width = font_size.0 * 0.6; // Approximate em width

            let mut glyphs = Vec::new();
            let mut position = 0.0f32;

            for (idx, ch) in text.char_indices() {
                glyphs.push(ShapedGlyph {
                    id: GlyphId(ch as u32),
                    position: point(px(position), px(0.)),
                    index: idx,
                    is_emoji: false,
                });
                position += em_width;
            }

            let font_id = runs.first().map(|r| r.font_id).unwrap_or(FontId(0));

            LineLayout {
                font_size,
                width: px(position),
                ascent: px(font_size.0 * metrics.ascent / metrics.units_per_em as f32),
                descent: px(font_size.0 * metrics.descent.abs() / metrics.units_per_em as f32),
                runs: vec![ShapedRun { font_id, glyphs }],
                len: text.len(),
            }
        }
    }
}

// Ensure WebTextSystem can be sent between threads
// (it uses RwLock internally for thread safety)
unsafe impl Send for WebTextSystem {}
unsafe impl Sync for WebTextSystem {}
