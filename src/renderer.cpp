#include "renderer.h"

#include <GL/glew.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

static const char* FULLSCREEN_VS = R"(#version 450 core
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID & 1) << 2, (gl_VertexID & 2) << 1) - 1.0;
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
)";

// --- Skybox: fullscreen, depth = 1.0 (drawn behind everything) -------------

static const char* SKY_VS = R"(#version 450 core
out vec3 v_view_dir;
uniform mat4 u_inv_view_proj;
uniform vec3 u_cam_pos;
void main() {
    vec2 p = vec2((gl_VertexID & 1) << 2, (gl_VertexID & 2) << 1) - 1.0;
    vec4 wp = u_inv_view_proj * vec4(p, 1.0, 1.0);
    v_view_dir = wp.xyz / wp.w - u_cam_pos;
    gl_Position = vec4(p, 1.0, 1.0);
}
)";

static const char* SKY_FS = R"(#version 450 core
in vec3 v_view_dir;
uniform vec3 u_sun_dir;
out vec4 frag;

vec3 sky(vec3 d, vec3 sun) {
    vec3 horizon = vec3(0.62, 0.72, 0.85);
    vec3 zenith  = vec3(0.20, 0.42, 0.78);
    vec3 ground  = vec3(0.18, 0.18, 0.20);
    vec3 col = mix(horizon, zenith, smoothstep(0.0, 0.7, d.y));
    col = mix(ground, col, smoothstep(-0.05, 0.05, d.y));
    float s = pow(max(dot(d, sun), 0.0), 256.0);
    col += vec3(1.00, 0.92, 0.78) * s * 1.6;
    float halo = pow(max(dot(d, sun), 0.0), 8.0);
    col += vec3(1.00, 0.85, 0.65) * halo * 0.08;
    return col;
}

void main() {
    vec3 d = normalize(v_view_dir);
    frag = vec4(sky(d, normalize(u_sun_dir)), 1.0);
}
)";

// --- Floor: a single quad with checkerboard + per-cell hash jitter ---------

static const char* FLOOR_VS = R"(#version 450 core
layout(location = 0) in vec2 in_xz;
uniform mat4  u_mvp;
uniform float u_y;
out vec3 v_world;
void main() {
    vec3 p = vec3(in_xz.x, u_y, in_xz.y);
    v_world = p;
    gl_Position = u_mvp * vec4(p, 1.0);
}
)";

static const char* FLOOR_FS = R"(#version 450 core
in vec3 v_world;
uniform float u_cell;
uniform float u_jitter;
uniform vec3  u_color_a;
uniform vec3  u_color_b;
uniform vec3  u_sun_dir;
out vec4 frag;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

void main() {
    vec2 cell = floor(v_world.xz / u_cell);
    float check = mod(cell.x + cell.y, 2.0);
    vec3 base = mix(u_color_a, u_color_b, check);
    float h = hash21(cell) - 0.5;
    base = clamp(base + vec3(h) * u_jitter, 0.0, 1.0);

    // Soft top-down lighting so the floor isn't flat-emissive
    vec3 n = vec3(0, 1, 0);
    float diff = clamp(dot(n, normalize(u_sun_dir)) * 0.5 + 0.5, 0.0, 1.0);
    base *= 0.55 + 0.45 * diff;
    frag = vec4(base, 1.0);
}
)";

// --- Particle depth (writes -eye_z into R32F + corrected gl_FragDepth) -----

static const char* PARTICLE_DEPTH_VS = R"(#version 450 core
layout(location = 0) in vec4 in_particle;
uniform mat4  u_mv;
uniform mat4  u_mvp;
uniform float u_radius_world;
uniform float u_proj11;
uniform float u_viewport_h;
out float v_eye_z;
void main() {
    vec4 pos_eye = u_mv * vec4(in_particle.xyz, 1.0);
    v_eye_z = pos_eye.z;
    float eye_neg = -pos_eye.z;
    gl_PointSize = max(1.0, u_radius_world * u_proj11 * u_viewport_h / eye_neg);
    gl_Position = u_mvp * vec4(in_particle.xyz, 1.0);
}
)";

static const char* PARTICLE_DEPTH_FS = R"(#version 450 core
in float v_eye_z;
uniform float u_sphere_radius;
uniform float u_proj22;
uniform float u_proj32;
out float frag_depth;
void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float sphere_z = sqrt(1.0 - r2);
    float eye_z = v_eye_z + u_sphere_radius * sphere_z;   // negative
    float ndc_z = (u_proj22 * eye_z + u_proj32) / (-eye_z);
    gl_FragDepth = ndc_z * 0.5 + 0.5;
    frag_depth = -eye_z; // store positive distance from camera
}
)";

// --- Particle thickness (additive Gaussian splat, no depth) ----------------

static const char* PARTICLE_THICK_VS = R"(#version 450 core
layout(location = 0) in vec4 in_particle;
uniform mat4  u_mv;
uniform mat4  u_mvp;
uniform float u_radius_world;
uniform float u_proj11;
uniform float u_viewport_h;
void main() {
    vec4 pos_eye = u_mv * vec4(in_particle.xyz, 1.0);
    float eye_neg = -pos_eye.z;
    gl_PointSize = max(1.0, u_radius_world * u_proj11 * u_viewport_h / eye_neg);
    gl_Position = u_mvp * vec4(in_particle.xyz, 1.0);
}
)";

static const char* PARTICLE_THICK_FS = R"(#version 450 core
uniform float u_thickness_scale;
out float frag;
void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    frag = exp(-r2 * 4.0) * u_thickness_scale;
}
)";

// --- Legacy debug: shaded sphere impostor ----------------------------------

static const char* PARTICLE_SPHERE_VS = R"(#version 450 core
layout(location = 0) in vec4 in_particle;
uniform mat4  u_mv;
uniform mat4  u_mvp;
uniform float u_radius_world;
uniform float u_proj11;
uniform float u_viewport_h;
out float v_density;
out float v_eye_z;
void main() {
    vec4 pos_eye = u_mv * vec4(in_particle.xyz, 1.0);
    v_eye_z = pos_eye.z;
    v_density = in_particle.w;
    float eye_neg = -pos_eye.z;
    gl_PointSize = max(1.0, u_radius_world * u_proj11 * u_viewport_h / eye_neg);
    gl_Position = u_mvp * vec4(in_particle.xyz, 1.0);
}
)";

static const char* PARTICLE_SPHERE_FS = R"(#version 450 core
in float v_density;
in float v_eye_z;
uniform float u_sphere_radius;
uniform float u_proj22;
uniform float u_proj32;
out vec4 frag;
void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float sphere_z = sqrt(1.0 - r2);
    vec3 n = vec3(uv.x, -uv.y, sphere_z);
    float eye_z = v_eye_z + u_sphere_radius * sphere_z;
    float ndc_z = (u_proj22 * eye_z + u_proj32) / (-eye_z);
    gl_FragDepth = ndc_z * 0.5 + 0.5;

    vec3 light = normalize(vec3(0.45, 0.75, 0.55));
    float diff = max(dot(n, light), 0.0);
    vec3 H = normalize(light + vec3(0, 0, 1));
    float spec = pow(max(dot(n, H), 0.0), 64.0);
    float fres = pow(1.0 - sphere_z, 3.0);

    float d = clamp((v_density - 0.8) / 0.4, 0.0, 1.0);
    vec3 base = mix(vec3(0.06, 0.28, 0.65), vec3(0.50, 0.82, 1.00), d);
    vec3 col = base * (0.15 + 0.85 * diff);
    col = mix(col, vec3(0.85, 0.95, 1.00), fres * 0.4);
    col += vec3(spec * 0.55);
    frag = vec4(col, 1.0);
}
)";

// --- Bilateral smoothing of eye-space depth --------------------------------

static const char* BILATERAL_FS = R"(#version 450 core
in vec2 v_uv;
uniform sampler2D u_input;
uniform vec2  u_texel;
uniform float u_sigma_s;
uniform float u_sigma_z;
uniform int   u_radius;
out float frag;
void main() {
    float center = texture(u_input, v_uv).r;
    if (center > 1.0e5) { frag = center; return; }

    float inv_2_s2 = 1.0 / (2.0 * u_sigma_s * u_sigma_s);
    float inv_2_z2 = 1.0 / (2.0 * u_sigma_z * u_sigma_z);
    float sum = 0.0;
    float wsum = 0.0;
    for (int dy = -u_radius; dy <= u_radius; ++dy) {
        for (int dx = -u_radius; dx <= u_radius; ++dx) {
            vec2 off = vec2(dx, dy) * u_texel;
            float d = texture(u_input, v_uv + off).r;
            if (d > 1.0e5) continue;
            float ws = exp(-float(dx*dx + dy*dy) * inv_2_s2);
            float dz = d - center;
            float wz = exp(-(dz*dz) * inv_2_z2);
            float w = ws * wz;
            sum  += w * d;
            wsum += w;
        }
    }
    frag = wsum > 0.0 ? sum / wsum : center;
}
)";

// --- Composite: SSF shading from smoothed depth + thickness + scene --------

static const char* COMPOSITE_FS = R"(#version 450 core
in vec2 v_uv;

uniform sampler2D u_smooth_depth;
uniform sampler2D u_thickness;
uniform sampler2D u_scene_color;

uniform mat4  u_inv_proj;
uniform mat4  u_inv_view;
uniform vec3  u_cam_pos;
uniform vec2  u_texel;
uniform float u_proj22;
uniform float u_proj32;

uniform vec3  u_water_color;
uniform vec3  u_absorption;
uniform float u_refraction;
uniform float u_fresnel_f0;
uniform float u_specular_power;
uniform float u_specular_intensity;
uniform vec3  u_sun_dir;
uniform int   u_debug_view;

out vec4 frag;

vec3 eye_pos_from_dist(vec2 uv, float dist) {
    float eye_z = -dist;
    float ndc_z = (u_proj22 * eye_z + u_proj32) / (-eye_z);
    vec3 ndc = vec3(uv * 2.0 - 1.0, ndc_z);
    vec4 ev = u_inv_proj * vec4(ndc, 1.0);
    return ev.xyz / ev.w;
}

vec3 sky(vec3 d, vec3 sun) {
    vec3 horizon = vec3(0.62, 0.72, 0.85);
    vec3 zenith  = vec3(0.20, 0.42, 0.78);
    vec3 ground  = vec3(0.18, 0.18, 0.20);
    vec3 col = mix(horizon, zenith, smoothstep(0.0, 0.7, d.y));
    col = mix(ground, col, smoothstep(-0.05, 0.05, d.y));
    float s = pow(max(dot(d, sun), 0.0), 256.0);
    col += vec3(1.00, 0.92, 0.78) * s * 1.6;
    return col;
}

void main() {
    float d_center = texture(u_smooth_depth, v_uv).r;
    vec3  scene    = texture(u_scene_color, v_uv).rgb;

    if (d_center > 1.0e5 || d_center <= 0.0) {
        frag = vec4(scene, 1.0);
        return;
    }

    vec3 ep = eye_pos_from_dist(v_uv, d_center);

    // Use the smaller (closer) of forward/backward differences to avoid
    // smearing normals across silhouettes.
    float dxp = texture(u_smooth_depth, v_uv + vec2(u_texel.x, 0)).r;
    float dxn = texture(u_smooth_depth, v_uv - vec2(u_texel.x, 0)).r;
    float dyp = texture(u_smooth_depth, v_uv + vec2(0, u_texel.y)).r;
    float dyn = texture(u_smooth_depth, v_uv - vec2(0, u_texel.y)).r;

    vec3 ddx;
    if (dxp < 1.0e5 && (dxn > 1.0e5 || abs(dxp - d_center) < abs(dxn - d_center)))
        ddx = eye_pos_from_dist(v_uv + vec2(u_texel.x, 0), dxp) - ep;
    else if (dxn < 1.0e5)
        ddx = ep - eye_pos_from_dist(v_uv - vec2(u_texel.x, 0), dxn);
    else
        ddx = vec3(1, 0, 0);

    vec3 ddy;
    if (dyp < 1.0e5 && (dyn > 1.0e5 || abs(dyp - d_center) < abs(dyn - d_center)))
        ddy = eye_pos_from_dist(v_uv + vec2(0, u_texel.y), dyp) - ep;
    else if (dyn < 1.0e5)
        ddy = ep - eye_pos_from_dist(v_uv - vec2(0, u_texel.y), dyn);
    else
        ddy = vec3(0, 1, 0);

    vec3 n_eye = normalize(cross(ddx, ddy));
    if (n_eye.z < 0.0) n_eye = -n_eye;

    float thickness = max(0.0, texture(u_thickness, v_uv).r);

    if (u_debug_view == 1) {
        float t = clamp(d_center / 30.0, 0.0, 1.0);
        frag = vec4(vec3(t), 1.0);
        return;
    }
    if (u_debug_view == 2) {
        frag = vec4(vec3(thickness), 1.0);
        return;
    }
    if (u_debug_view == 3) {
        frag = vec4(n_eye * 0.5 + 0.5, 1.0);
        return;
    }

    vec2 refr_uv = clamp(v_uv + n_eye.xy * u_refraction, 0.0, 1.0);
    vec3 refracted = texture(u_scene_color, refr_uv).rgb;
    vec3 absorb = exp(-u_absorption * thickness);
    refracted = refracted * absorb + u_water_color * (1.0 - absorb);

    vec3 n_world  = normalize(mat3(u_inv_view) * n_eye);
    vec3 v_world  = normalize(mat3(u_inv_view) * normalize(ep));
    vec3 r        = reflect(v_world, n_world);
    vec3 sun      = normalize(u_sun_dir);
    vec3 reflcol  = sky(r, sun);

    float ndotv  = max(dot(n_eye, vec3(0, 0, 1)), 0.0);
    float fresnel = u_fresnel_f0 + (1.0 - u_fresnel_f0) * pow(1.0 - ndotv, 5.0);

    vec3 col = mix(refracted, reflcol, fresnel);

    vec3 H = normalize(sun - v_world);
    float spec = pow(max(dot(n_world, H), 0.0), u_specular_power);
    col += vec3(spec * u_specular_intensity);

    frag = vec4(col, 1.0);
}
)";

// ---------------------------------------------------------------------------
// GL helpers
// ---------------------------------------------------------------------------

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "shader compile error: %s\n", log);
        std::exit(1);
    }
    return s;
}

static GLuint link_program(const char* vs_src, const char* fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "program link error: %s\n", log);
        std::exit(1);
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// ---------------------------------------------------------------------------
// Renderer impl
// ---------------------------------------------------------------------------

struct Renderer::Impl {
    int   w = 0, h = 0;
    float particle_radius = 0.05f;

    // FBOs
    GLuint fbo_scene     = 0;
    GLuint fbo_depth     = 0;
    GLuint fbo_smooth_a  = 0;
    GLuint fbo_smooth_b  = 0;
    GLuint fbo_thickness = 0;

    // Color attachments / sampled textures
    GLuint tex_scene_color = 0;
    GLuint tex_depth_eye   = 0; // R32F
    GLuint tex_smooth_a    = 0; // R32F
    GLuint tex_smooth_b    = 0; // R32F
    GLuint tex_thickness   = 0; // R16F

    // Depth attachments
    GLuint rb_scene_depth     = 0;
    GLuint rb_particle_depth  = 0;

    // Programs
    GLuint prog_sky        = 0;
    GLuint prog_floor      = 0;
    GLuint prog_pdepth     = 0;
    GLuint prog_pthick     = 0;
    GLuint prog_psphere    = 0;
    GLuint prog_bilateral  = 0;
    GLuint prog_composite  = 0;

    // Geometry
    GLuint vao_empty = 0; // for fullscreen draws (no buffers)
    GLuint vao_floor = 0;
    GLuint vbo_floor = 0;
    GLuint ebo_floor = 0;
};

Renderer::Renderer()  { impl_ = new Impl(); }
Renderer::~Renderer() { delete impl_; }

static GLuint make_color_tex(int w, int h, GLenum internal, GLenum format, GLenum type, GLint filter) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return t;
}

static GLuint make_depth_rb(int w, int h) {
    GLuint rb = 0;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    return rb;
}

static void check_fbo(const char* tag) {
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "FBO incomplete (%s): 0x%x\n", tag, st);
        std::exit(1);
    }
}

void Renderer::init(const RendererInit& info) {
    Impl& s = *impl_;
    s.w = info.width;
    s.h = info.height;
    s.particle_radius = info.particle_radius;

    // --- Programs --------------------------------------------------------
    s.prog_sky       = link_program(SKY_VS,             SKY_FS);
    s.prog_floor     = link_program(FLOOR_VS,           FLOOR_FS);
    s.prog_pdepth    = link_program(PARTICLE_DEPTH_VS,  PARTICLE_DEPTH_FS);
    s.prog_pthick    = link_program(PARTICLE_THICK_VS,  PARTICLE_THICK_FS);
    s.prog_psphere   = link_program(PARTICLE_SPHERE_VS, PARTICLE_SPHERE_FS);
    s.prog_bilateral = link_program(FULLSCREEN_VS,      BILATERAL_FS);
    s.prog_composite = link_program(FULLSCREEN_VS,      COMPOSITE_FS);

    // --- Empty VAO for fullscreen passes ---------------------------------
    glGenVertexArrays(1, &s.vao_empty);

    // --- Floor quad ------------------------------------------------------
    {
        const float extent = 80.0f;
        float quad[] = {
            -extent, -extent,
             extent, -extent,
             extent,  extent,
            -extent,  extent,
        };
        unsigned int idx[] = {0, 1, 2, 0, 2, 3};
        glGenVertexArrays(1, &s.vao_floor);
        glGenBuffers(1, &s.vbo_floor);
        glGenBuffers(1, &s.ebo_floor);
        glBindVertexArray(s.vao_floor);
        glBindBuffer(GL_ARRAY_BUFFER, s.vbo_floor);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s.ebo_floor);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, nullptr);
        glBindVertexArray(0);
    }

    // --- Render targets --------------------------------------------------
    int w = s.w, h = s.h;

    s.tex_scene_color = make_color_tex(w, h, GL_RGBA8,  GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR);
    s.tex_depth_eye   = make_color_tex(w, h, GL_R32F,   GL_RED,  GL_FLOAT,        GL_NEAREST);
    s.tex_smooth_a    = make_color_tex(w, h, GL_R32F,   GL_RED,  GL_FLOAT,        GL_NEAREST);
    s.tex_smooth_b    = make_color_tex(w, h, GL_R32F,   GL_RED,  GL_FLOAT,        GL_NEAREST);
    s.tex_thickness   = make_color_tex(w, h, GL_R16F,   GL_RED,  GL_FLOAT,        GL_LINEAR);

    s.rb_scene_depth     = make_depth_rb(w, h);
    s.rb_particle_depth  = make_depth_rb(w, h);

    auto make_fbo = [](GLuint color, GLuint depth_rb, const char* tag) {
        GLuint fbo = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        if (depth_rb) {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                       GL_RENDERBUFFER, depth_rb);
        }
        GLenum bufs[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, bufs);
        check_fbo(tag);
        return fbo;
    };

    s.fbo_scene     = make_fbo(s.tex_scene_color, s.rb_scene_depth,    "scene");
    s.fbo_depth     = make_fbo(s.tex_depth_eye,   s.rb_particle_depth, "particle-depth");
    s.fbo_smooth_a  = make_fbo(s.tex_smooth_a,    0,                   "smooth-a");
    s.fbo_smooth_b  = make_fbo(s.tex_smooth_b,    0,                   "smooth-b");
    s.fbo_thickness = make_fbo(s.tex_thickness,   0,                   "thickness");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::shutdown() {
    Impl& s = *impl_;
    GLuint fbos[] = { s.fbo_scene, s.fbo_depth, s.fbo_smooth_a, s.fbo_smooth_b, s.fbo_thickness };
    glDeleteFramebuffers(5, fbos);
    GLuint texs[] = { s.tex_scene_color, s.tex_depth_eye, s.tex_smooth_a, s.tex_smooth_b, s.tex_thickness };
    glDeleteTextures(5, texs);
    GLuint rbs[] = { s.rb_scene_depth, s.rb_particle_depth };
    glDeleteRenderbuffers(2, rbs);
    GLuint progs[] = { s.prog_sky, s.prog_floor, s.prog_pdepth, s.prog_pthick,
                       s.prog_psphere, s.prog_bilateral, s.prog_composite };
    for (GLuint p : progs) if (p) glDeleteProgram(p);
    if (s.vao_empty) glDeleteVertexArrays(1, &s.vao_empty);
    if (s.vao_floor) glDeleteVertexArrays(1, &s.vao_floor);
    if (s.vbo_floor) glDeleteBuffers(1, &s.vbo_floor);
    if (s.ebo_floor) glDeleteBuffers(1, &s.ebo_floor);
    std::memset(&s, 0, sizeof(Impl));
}

// Draws sky + floor into the scene FBO. Leaves fbo_scene bound on exit.
static void render_scene(Renderer::Impl& s, const glm::mat4& view, const glm::mat4& proj,
                          const glm::vec3& cam_pos, const RenderSettings& set,
                          const float bg_color[3]) {
    glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_scene);
    glViewport(0, 0, s.w, s.h);
    glClearColor(bg_color[0], bg_color[1], bg_color[2], 1.0f);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_BLEND);

    if (set.show_skybox) {
        glDepthMask(GL_FALSE);
        glDepthFunc(GL_LEQUAL);
        glUseProgram(s.prog_sky);
        glm::mat4 ivp = glm::inverse(proj * view);
        glUniformMatrix4fv(glGetUniformLocation(s.prog_sky, "u_inv_view_proj"),
                            1, GL_FALSE, glm::value_ptr(ivp));
        glUniform3fv(glGetUniformLocation(s.prog_sky, "u_cam_pos"), 1, glm::value_ptr(cam_pos));
        glUniform3fv(glGetUniformLocation(s.prog_sky, "u_sun_dir"), 1, set.sun_dir);
        glBindVertexArray(s.vao_empty);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
    }

    if (set.show_floor) {
        glEnable(GL_DEPTH_TEST);
        glUseProgram(s.prog_floor);
        glm::mat4 mvp = proj * view;
        glUniformMatrix4fv(glGetUniformLocation(s.prog_floor, "u_mvp"),
                            1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1f(glGetUniformLocation(s.prog_floor, "u_y"),       set.floor_y);
        glUniform1f(glGetUniformLocation(s.prog_floor, "u_cell"),    set.floor_cell);
        glUniform1f(glGetUniformLocation(s.prog_floor, "u_jitter"),  set.floor_jitter);
        glUniform3fv(glGetUniformLocation(s.prog_floor, "u_color_a"), 1, set.floor_color_a);
        glUniform3fv(glGetUniformLocation(s.prog_floor, "u_color_b"), 1, set.floor_color_b);
        glUniform3fv(glGetUniformLocation(s.prog_floor, "u_sun_dir"), 1, set.sun_dir);
        glBindVertexArray(s.vao_floor);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }
}

void Renderer::render(unsigned int particle_vao, int particle_count,
                       const glm::mat4& view, const glm::mat4& proj,
                       const glm::vec3& cam_pos, const RenderSettings& set,
                       const float bg_color[3]) {
    Impl& s = *impl_;
    const int w = s.w, h = s.h;

    glm::mat4 mvp = proj * view;
    const float proj11 = proj[1][1];
    const float proj22 = proj[2][2];
    const float proj32 = proj[3][2];

    // ---- Pass 1: scene (sky + floor) into fbo_scene --------------------
    render_scene(s, view, proj, cam_pos, set, bg_color);

    if (set.mode == RenderSettings::Spheres) {
        // Render sphere impostors directly on top of scene FB, then blit to default.
        glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_scene);
        glViewport(0, 0, w, h);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glUseProgram(s.prog_psphere);
        glUniformMatrix4fv(glGetUniformLocation(s.prog_psphere, "u_mv"),  1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(s.prog_psphere, "u_mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1f(glGetUniformLocation(s.prog_psphere, "u_radius_world"),  s.particle_radius);
        glUniform1f(glGetUniformLocation(s.prog_psphere, "u_proj11"),        proj11);
        glUniform1f(glGetUniformLocation(s.prog_psphere, "u_viewport_h"),    static_cast<float>(h));
        glUniform1f(glGetUniformLocation(s.prog_psphere, "u_sphere_radius"), s.particle_radius);
        glUniform1f(glGetUniformLocation(s.prog_psphere, "u_proj22"),        proj22);
        glUniform1f(glGetUniformLocation(s.prog_psphere, "u_proj32"),        proj32);
        glBindVertexArray(particle_vao);
        glDrawArrays(GL_POINTS, 0, particle_count);
        glBindVertexArray(0);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, s.fbo_scene);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, w, h);
        return;
    }

    // ---- Pass 2: particle eye-space depth ------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_depth);
    glViewport(0, 0, w, h);
    {
        const float clear_far[4] = {1.0e6f, 0, 0, 0};
        glClearBufferfv(GL_COLOR, 0, clear_far);
    }
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    glUseProgram(s.prog_pdepth);
    glUniformMatrix4fv(glGetUniformLocation(s.prog_pdepth, "u_mv"),  1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(s.prog_pdepth, "u_mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(glGetUniformLocation(s.prog_pdepth, "u_radius_world"),  s.particle_radius);
    glUniform1f(glGetUniformLocation(s.prog_pdepth, "u_proj11"),        proj11);
    glUniform1f(glGetUniformLocation(s.prog_pdepth, "u_viewport_h"),    static_cast<float>(h));
    glUniform1f(glGetUniformLocation(s.prog_pdepth, "u_sphere_radius"), s.particle_radius);
    glUniform1f(glGetUniformLocation(s.prog_pdepth, "u_proj22"),        proj22);
    glUniform1f(glGetUniformLocation(s.prog_pdepth, "u_proj32"),        proj32);
    glBindVertexArray(particle_vao);
    glDrawArrays(GL_POINTS, 0, particle_count);

    // ---- Pass 3: bilateral smoothing (ping-pong) -----------------------
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glUseProgram(s.prog_bilateral);
    glUniform2f(glGetUniformLocation(s.prog_bilateral, "u_texel"),
                 1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h));
    glUniform1f(glGetUniformLocation(s.prog_bilateral, "u_sigma_s"), set.sigma_spatial);
    glUniform1f(glGetUniformLocation(s.prog_bilateral, "u_sigma_z"), set.sigma_depth);
    glUniform1i(glGetUniformLocation(s.prog_bilateral, "u_radius"),  set.smooth_radius);
    glUniform1i(glGetUniformLocation(s.prog_bilateral, "u_input"),   0);
    glActiveTexture(GL_TEXTURE0);

    GLuint src_tex = s.tex_depth_eye;
    int    iters   = set.smooth_iters > 0 ? set.smooth_iters : 0;
    for (int i = 0; i < iters; ++i) {
        bool to_a = (i % 2 == 0);
        GLuint dst_fbo = to_a ? s.fbo_smooth_a : s.fbo_smooth_b;
        GLuint dst_tex = to_a ? s.tex_smooth_a : s.tex_smooth_b;
        glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo);
        glViewport(0, 0, w, h);
        glBindTexture(GL_TEXTURE_2D, src_tex);
        glBindVertexArray(s.vao_empty);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        src_tex = dst_tex;
    }
    GLuint smoothed = src_tex;

    // ---- Pass 4: thickness (additive) ----------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, s.fbo_thickness);
    glViewport(0, 0, w, h);
    {
        const float clear_zero[4] = {0, 0, 0, 0};
        glClearBufferfv(GL_COLOR, 0, clear_zero);
    }
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    glUseProgram(s.prog_pthick);
    glUniformMatrix4fv(glGetUniformLocation(s.prog_pthick, "u_mv"),  1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(s.prog_pthick, "u_mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(glGetUniformLocation(s.prog_pthick, "u_radius_world"),
                 s.particle_radius * set.thickness_radius_mul);
    glUniform1f(glGetUniformLocation(s.prog_pthick, "u_proj11"),     proj11);
    glUniform1f(glGetUniformLocation(s.prog_pthick, "u_viewport_h"), static_cast<float>(h));
    glUniform1f(glGetUniformLocation(s.prog_pthick, "u_thickness_scale"), set.thickness_scale);
    glBindVertexArray(particle_vao);
    glDrawArrays(GL_POINTS, 0, particle_count);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

    // ---- Pass 5: composite to default FB -------------------------------
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, w, h);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glUseProgram(s.prog_composite);

    glUniform1i(glGetUniformLocation(s.prog_composite, "u_smooth_depth"), 0);
    glUniform1i(glGetUniformLocation(s.prog_composite, "u_thickness"),    1);
    glUniform1i(glGetUniformLocation(s.prog_composite, "u_scene_color"),  2);

    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, smoothed);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, s.tex_thickness);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, s.tex_scene_color);

    glm::mat4 inv_proj = glm::inverse(proj);
    glm::mat4 inv_view = glm::inverse(view);
    glUniformMatrix4fv(glGetUniformLocation(s.prog_composite, "u_inv_proj"), 1, GL_FALSE, glm::value_ptr(inv_proj));
    glUniformMatrix4fv(glGetUniformLocation(s.prog_composite, "u_inv_view"), 1, GL_FALSE, glm::value_ptr(inv_view));
    glUniform3fv(glGetUniformLocation(s.prog_composite, "u_cam_pos"), 1, glm::value_ptr(cam_pos));
    glUniform2f(glGetUniformLocation(s.prog_composite, "u_texel"),
                 1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h));
    glUniform1f(glGetUniformLocation(s.prog_composite, "u_proj22"), proj22);
    glUniform1f(glGetUniformLocation(s.prog_composite, "u_proj32"), proj32);

    glUniform3fv(glGetUniformLocation(s.prog_composite, "u_water_color"), 1, set.water_color);
    glUniform3fv(glGetUniformLocation(s.prog_composite, "u_absorption"),  1, set.absorption);
    glUniform1f(glGetUniformLocation(s.prog_composite, "u_refraction"),         set.refraction_strength);
    glUniform1f(glGetUniformLocation(s.prog_composite, "u_fresnel_f0"),         set.fresnel_f0);
    glUniform1f(glGetUniformLocation(s.prog_composite, "u_specular_power"),     set.specular_power);
    glUniform1f(glGetUniformLocation(s.prog_composite, "u_specular_intensity"), set.specular_intensity);
    glUniform3fv(glGetUniformLocation(s.prog_composite, "u_sun_dir"), 1, set.sun_dir);
    glUniform1i(glGetUniformLocation(s.prog_composite, "u_debug_view"), set.debug_view);

    glBindVertexArray(s.vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
}
