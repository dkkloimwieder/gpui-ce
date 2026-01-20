// WebRenderer shaders for GPUI WASM
// Simplified from BladeRenderer shaders.wgsl

// === Common Types === //

struct GlobalParams {
    viewport_size: vec2<f32>,
    premultiplied_alpha: u32,
    pad: u32,
}

var<uniform> globals: GlobalParams;

struct Bounds {
    origin_x: f32,
    origin_y: f32,
    size_width: f32,
    size_height: f32,
}

struct Corners {
    top_left: f32,
    top_right: f32,
    bottom_right: f32,
    bottom_left: f32,
}

struct Edges {
    top: f32,
    right: f32,
    bottom: f32,
    left: f32,
}

struct Hsla {
    h: f32,
    s: f32,
    l: f32,
    a: f32,
}

struct LinearColorStop {
    color: Hsla,
    percentage: f32,
}

struct Background {
    tag: u32,
    color_space: u32,
    solid: Hsla,
    gradient_angle_or_pattern_height: f32,
    colors: array<LinearColorStop, 2>,
    pad: u32,
}

// === Helper Functions === //

fn to_device_position(unit_vertex: vec2<f32>, bounds: Bounds) -> vec4<f32> {
    let origin = vec2<f32>(bounds.origin_x, bounds.origin_y);
    let size = vec2<f32>(bounds.size_width, bounds.size_height);
    let position = unit_vertex * size + origin;
    let device_position = position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    return vec4<f32>(device_position, 0.0, 1.0);
}

fn distance_from_clip_rect(unit_vertex: vec2<f32>, bounds: Bounds, clip_bounds: Bounds) -> vec4<f32> {
    let origin = vec2<f32>(bounds.origin_x, bounds.origin_y);
    let size = vec2<f32>(bounds.size_width, bounds.size_height);
    let clip_origin = vec2<f32>(clip_bounds.origin_x, clip_bounds.origin_y);
    let clip_size = vec2<f32>(clip_bounds.size_width, clip_bounds.size_height);
    let position = unit_vertex * size + origin;
    let tl = position - clip_origin;
    let br = clip_origin + clip_size - position;
    return vec4<f32>(tl.x, br.x, tl.y, br.y);
}

/// Hsla to linear RGBA conversion.
fn hsla_to_rgba(hsla: Hsla) -> vec4<f32> {
    let h = hsla.h * 6.0;
    let s = hsla.s;
    let l = hsla.l;
    let a = hsla.a;

    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs(h % 2.0 - 1.0));
    let m = l - c / 2.0;
    var color = vec3<f32>(m);

    if (h >= 0.0 && h < 1.0) {
        color.r += c;
        color.g += x;
    } else if (h >= 1.0 && h < 2.0) {
        color.r += x;
        color.g += c;
    } else if (h >= 2.0 && h < 3.0) {
        color.g += c;
        color.b += x;
    } else if (h >= 3.0 && h < 4.0) {
        color.g += x;
        color.b += c;
    } else if (h >= 4.0 && h < 5.0) {
        color.r += x;
        color.b += c;
    } else {
        color.r += c;
        color.b += x;
    }

    return vec4<f32>(color, a);
}

// Selects corner radius based on quadrant.
fn pick_corner_radius(center_to_point: vec2<f32>, radii: Corners) -> f32 {
    if (center_to_point.x < 0.0) {
        if (center_to_point.y < 0.0) {
            return radii.top_left;
        } else {
            return radii.bottom_left;
        }
    } else {
        if (center_to_point.y < 0.0) {
            return radii.top_right;
        } else {
            return radii.bottom_right;
        }
    }
}

fn quad_sdf_impl(corner_center_to_point: vec2<f32>, corner_radius: f32) -> f32 {
    if (corner_radius == 0.0) {
        return max(corner_center_to_point.x, corner_center_to_point.y);
    } else {
        let signed_distance_to_inset_quad =
            length(max(vec2<f32>(0.0), corner_center_to_point)) +
            min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));
        return signed_distance_to_inset_quad - corner_radius;
    }
}

fn blend_color(color: vec4<f32>, alpha_factor: f32) -> vec4<f32> {
    let alpha = color.a * alpha_factor;
    let multiplier = select(1.0, alpha, globals.premultiplied_alpha != 0u);
    return vec4<f32>(color.rgb * multiplier, alpha);
}

fn over(below: vec4<f32>, above: vec4<f32>) -> vec4<f32> {
    let alpha = above.a + below.a * (1.0 - above.a);
    let color = (above.rgb * above.a + below.rgb * below.a * (1.0 - above.a)) / alpha;
    return vec4<f32>(color, alpha);
}

// === Quad Shader === //

struct Quad {
    order: u32,
    border_style: u32,
    bounds: Bounds,
    content_mask: Bounds,
    background: Background,
    border_color: Hsla,
    corner_radii: Corners,
    border_widths: Edges,
}

var<storage, read> b_quads: array<Quad>;

struct QuadVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) background_color: vec4<f32>,
    @location(1) @interpolate(flat) border_color: vec4<f32>,
    @location(2) @interpolate(flat) quad_id: u32,
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_quad(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> QuadVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let quad = b_quads[instance_id];

    var out = QuadVarying();
    out.position = to_device_position(unit_vertex, quad.bounds);
    // For now, only support solid colors (tag == 0)
    out.background_color = hsla_to_rgba(quad.background.solid);
    out.border_color = hsla_to_rgba(quad.border_color);
    out.quad_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, quad.bounds, quad.content_mask);
    return out;
}

@fragment
fn fs_quad(input: QuadVarying) -> @location(0) vec4<f32> {
    // Alpha clip first
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let quad = b_quads[input.quad_id];
    let background_color = input.background_color;

    let unrounded = quad.corner_radii.top_left == 0.0 &&
        quad.corner_radii.bottom_left == 0.0 &&
        quad.corner_radii.top_right == 0.0 &&
        quad.corner_radii.bottom_right == 0.0;

    // Fast path for unrounded quads without borders
    if (quad.border_widths.top == 0.0 &&
            quad.border_widths.left == 0.0 &&
            quad.border_widths.right == 0.0 &&
            quad.border_widths.bottom == 0.0 &&
            unrounded) {
        return blend_color(background_color, 1.0);
    }

    let size = vec2<f32>(quad.bounds.size_width, quad.bounds.size_height);
    let half_size = size / 2.0;
    let point = input.position.xy - vec2<f32>(quad.bounds.origin_x, quad.bounds.origin_y);
    let center_to_point = point - half_size;

    let antialias_threshold = 0.5;
    let corner_radius = pick_corner_radius(center_to_point, quad.corner_radii);

    let border = vec2<f32>(
        select(quad.border_widths.right, quad.border_widths.left, center_to_point.x < 0.0),
        select(quad.border_widths.bottom, quad.border_widths.top, center_to_point.y < 0.0));

    let reduced_border =
        vec2<f32>(select(border.x, -antialias_threshold, border.x == 0.0),
                  select(border.y, -antialias_threshold, border.y == 0.0));

    let corner_to_point = abs(center_to_point) - half_size;
    let corner_center_to_point = corner_to_point + corner_radius;

    let is_near_rounded_corner =
            corner_center_to_point.x >= 0 &&
            corner_center_to_point.y >= 0;

    let straight_border_inner_corner_to_point = corner_to_point + reduced_border;

    let is_within_inner_straight_border =
        straight_border_inner_corner_to_point.x < -antialias_threshold &&
        straight_border_inner_corner_to_point.y < -antialias_threshold;

    // Fast path for background
    if (is_within_inner_straight_border && !is_near_rounded_corner) {
        return blend_color(background_color, 1.0);
    }

    let outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius);

    var inner_sdf = 0.0;
    if (corner_center_to_point.x <= 0 || corner_center_to_point.y <= 0) {
        inner_sdf = -max(straight_border_inner_corner_to_point.x,
                         straight_border_inner_corner_to_point.y);
    } else {
        inner_sdf = -(outer_sdf + min(reduced_border.x, reduced_border.y));
    }

    let border_sdf = max(inner_sdf, outer_sdf);

    var color = background_color;
    if (border_sdf < antialias_threshold) {
        let blended_border = over(background_color, input.border_color);
        color = mix(background_color, blended_border,
                    saturate(antialias_threshold - inner_sdf));
    }

    return blend_color(color, saturate(antialias_threshold - outer_sdf));
}

// === Atlas Types === //

struct AtlasTextureId {
    index: u32,
    kind: u32,
}

struct AtlasBounds {
    origin: vec2<i32>,
    size: vec2<i32>,
}

struct AtlasTile {
    texture_id: AtlasTextureId,
    tile_id: u32,
    padding: u32,
    bounds: AtlasBounds,
}

// Sprite texture and sampler
var t_sprite: texture_2d<f32>;
var s_sprite: sampler;

fn to_tile_position(unit_vertex: vec2<f32>, tile: AtlasTile) -> vec2<f32> {
    let atlas_size = vec2<f32>(textureDimensions(t_sprite, 0));
    return (vec2<f32>(tile.bounds.origin) + unit_vertex * vec2<f32>(tile.bounds.size)) / atlas_size;
}

// === Transformation Matrix === //

struct TransformationMatrix {
    rotation_scale: mat2x2<f32>,
    translation: vec2<f32>,
}

// === Monochrome Sprite Shader === //

struct MonochromeSprite {
    order: u32,
    pad: u32,
    bounds: Bounds,
    content_mask: Bounds,
    color: Hsla,
    tile: AtlasTile,
    transformation: TransformationMatrix,
}

var<storage, read> b_mono_sprites: array<MonochromeSprite>;

struct MonoSpriteVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tile_position: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) clip_distances: vec4<f32>,
}

@vertex
fn vs_mono_sprite(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> MonoSpriteVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = b_mono_sprites[instance_id];

    var out = MonoSpriteVarying();
    out.position = to_device_position(unit_vertex, sprite.bounds);
    out.tile_position = to_tile_position(unit_vertex, sprite.tile);
    out.color = hsla_to_rgba(sprite.color);
    out.clip_distances = distance_from_clip_rect(unit_vertex, sprite.bounds, sprite.content_mask);
    return out;
}

@fragment
fn fs_mono_sprite(input: MonoSpriteVarying) -> @location(0) vec4<f32> {
    // Sample grayscale value from atlas first (must be in uniform control flow)
    let sample = textureSample(t_sprite, s_sprite, input.tile_position).r;

    // Alpha clip after texture sample
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    // Apply color tint
    return blend_color(input.color, sample * input.color.a);
}

// === Polychrome Sprite Shader === //

struct PolychromeSprite {
    order: u32,
    pad: u32,
    grayscale: u32,
    opacity: f32,
    bounds: Bounds,
    content_mask: Bounds,
    corner_radii: Corners,
    tile: AtlasTile,
}

var<storage, read> b_poly_sprites: array<PolychromeSprite>;

struct PolySpriteVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tile_position: vec2<f32>,
    @location(1) @interpolate(flat) sprite_id: u32,
    @location(2) clip_distances: vec4<f32>,
}

@vertex
fn vs_poly_sprite(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> PolySpriteVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let sprite = b_poly_sprites[instance_id];

    var out = PolySpriteVarying();
    out.position = to_device_position(unit_vertex, sprite.bounds);
    out.tile_position = to_tile_position(unit_vertex, sprite.tile);
    out.sprite_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, sprite.bounds, sprite.content_mask);
    return out;
}

const GRAYSCALE_FACTORS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

@fragment
fn fs_poly_sprite(input: PolySpriteVarying) -> @location(0) vec4<f32> {
    // Sample texture first (must be in uniform control flow)
    let sample = textureSample(t_sprite, s_sprite, input.tile_position);
    let sprite = b_poly_sprites[input.sprite_id];

    // Alpha clip after texture sample
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    var color = sample;
    // Apply grayscale if requested
    if ((sprite.grayscale & 0xFFu) != 0u) {
        let grayscale = dot(color.rgb, GRAYSCALE_FACTORS);
        color = vec4<f32>(vec3<f32>(grayscale), sample.a);
    }

    return blend_color(color, sprite.opacity);
}

// === Shadow Shader === //

const M_PI_F: f32 = 3.1415926;

struct Shadow {
    order: u32,
    blur_radius: f32,
    bounds: Bounds,
    corner_radii: Corners,
    content_mask: Bounds,
    color: Hsla,
}

var<storage, read> b_shadows: array<Shadow>;

struct ShadowVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) shadow_id: u32,
    @location(2) clip_distances: vec4<f32>,
}

// A standard gaussian function, used for weighting samples
fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * M_PI_F) * sigma);
}

// This approximates the error function, needed for the gaussian integral
fn erf(v: vec2<f32>) -> vec2<f32> {
    let s = sign(v);
    let a = abs(v);
    let r1 = 1.0 + (0.278393 + (0.230389 + (0.000972 + 0.078108 * a) * a) * a) * a;
    let r2 = r1 * r1;
    return s - s / (r2 * r2);
}

fn blur_along_x(x: f32, y: f32, sigma: f32, corner: f32, half_size: vec2<f32>) -> f32 {
    let delta = min(half_size.y - corner - abs(y), 0.0);
    let curved = half_size.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
    let integral = 0.5 + 0.5 * erf((x + vec2<f32>(-curved, curved)) * (sqrt(0.5) / sigma));
    return integral.y - integral.x;
}

@vertex
fn vs_shadow(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> ShadowVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    var shadow = b_shadows[instance_id];

    // Expand bounds by blur margin
    let margin = 3.0 * shadow.blur_radius;
    shadow.bounds.origin_x -= margin;
    shadow.bounds.origin_y -= margin;
    shadow.bounds.size_width += 2.0 * margin;
    shadow.bounds.size_height += 2.0 * margin;

    var out = ShadowVarying();
    out.position = to_device_position(unit_vertex, shadow.bounds);
    out.color = hsla_to_rgba(shadow.color);
    out.shadow_id = instance_id;
    out.clip_distances = distance_from_clip_rect(unit_vertex, shadow.bounds, shadow.content_mask);
    return out;
}

@fragment
fn fs_shadow(input: ShadowVarying) -> @location(0) vec4<f32> {
    // Alpha clip first
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    let shadow = b_shadows[input.shadow_id];
    let half_size = vec2<f32>(shadow.bounds.size_width, shadow.bounds.size_height) / 2.0;
    let center = vec2<f32>(shadow.bounds.origin_x, shadow.bounds.origin_y) + half_size;
    let center_to_point = input.position.xy - center;

    let corner_radius = pick_corner_radius(center_to_point, shadow.corner_radii);

    // The signal is only non-zero in a limited range, so don't waste samples
    let low = center_to_point.y - half_size.y;
    let high = center_to_point.y + half_size.y;
    let start = clamp(-3.0 * shadow.blur_radius, low, high);
    let end = clamp(3.0 * shadow.blur_radius, low, high);

    // Accumulate samples (we can get away with surprisingly few samples)
    let step = (end - start) / 4.0;
    var y = start + step * 0.5;
    var alpha = 0.0;
    for (var i = 0; i < 4; i += 1) {
        let blur = blur_along_x(center_to_point.x, center_to_point.y - y,
            shadow.blur_radius, corner_radius, half_size);
        alpha += blur * gaussian(y, shadow.blur_radius) * step;
        y += step;
    }

    // Use blur alpha directly; color's alpha controls shadow darkness, not coverage
    return vec4<f32>(input.color.rgb * input.color.a, alpha);
}

// === Path Shader === //

// Path vertex structure - matches Rust PathVertex layout
// Each vertex contains position, st coords for bezier AA, and content mask
struct PathVertex {
    xy_position_x: f32,
    xy_position_y: f32,
    st_position_x: f32,
    st_position_y: f32,
    content_mask: Bounds,
    color: Hsla,
}

var<storage, read> b_path_vertices: array<PathVertex>;

struct PathVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) st_position: vec2<f32>,
    @location(1) @interpolate(flat) color: vec4<f32>,
    @location(2) clip_distances: vec4<f32>,
}

@vertex
fn vs_path(@builtin(vertex_index) vertex_id: u32) -> PathVarying {
    let v = b_path_vertices[vertex_id];
    let xy_position = vec2<f32>(v.xy_position_x, v.xy_position_y);

    // Convert to device coordinates
    let device_position = xy_position / globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);

    var out = PathVarying();
    out.position = vec4<f32>(device_position, 0.0, 1.0);
    out.st_position = vec2<f32>(v.st_position_x, v.st_position_y);
    out.color = hsla_to_rgba(v.color);

    // Clip distances for content mask
    let clip_origin = vec2<f32>(v.content_mask.origin_x, v.content_mask.origin_y);
    let clip_size = vec2<f32>(v.content_mask.size_width, v.content_mask.size_height);
    let tl = xy_position - clip_origin;
    let br = clip_origin + clip_size - xy_position;
    out.clip_distances = vec4<f32>(tl.x, br.x, tl.y, br.y);

    return out;
}

@fragment
fn fs_path(input: PathVarying) -> @location(0) vec4<f32> {
    // Alpha clip
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }

    // Quadratic bezier anti-aliasing using st coordinates
    let dx = dpdx(input.st_position);
    let dy = dpdy(input.st_position);

    var alpha: f32;
    if (length(vec2<f32>(dx.x, dy.x)) < 0.001) {
        // If the gradient is too small, return a solid color
        alpha = 1.0;
    } else {
        let gradient = 2.0 * input.st_position.xx * vec2<f32>(dx.x, dy.x) - vec2<f32>(dx.y, dy.y);
        let f = input.st_position.x * input.st_position.x - input.st_position.y;
        let distance = f / length(gradient);
        alpha = saturate(0.5 - distance);
    }

    return blend_color(input.color, alpha);
}
