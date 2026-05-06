#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "kernel.cuh"
#include "scene_desc.h"

namespace fs = std::filesystem;

struct FrameSample {
    double t;
    float sim_ms;
    float render_ms;
};

struct FrameTimingWindow {
    static constexpr double kWindowSeconds = 1.0;
    std::deque<FrameSample> samples;

    void push(double now, float sim_ms, float render_ms) {
        samples.push_back({now, sim_ms, render_ms});
        while (!samples.empty() && now - samples.front().t > kWindowSeconds) {
            samples.pop_front();
        }
    }

    void summarize(float& avg_sim, float& max_sim, float& avg_render,
                   float& max_render) const {
        avg_sim = max_sim = avg_render = max_render = 0.0f;
        if (samples.empty()) {
            return;
        }
        double sum_sim = 0.0, sum_render = 0.0;
        for (const FrameSample& s : samples) {
            sum_sim += s.sim_ms;
            sum_render += s.render_ms;
            if (s.sim_ms > max_sim) max_sim = s.sim_ms;
            if (s.render_ms > max_render) max_render = s.render_ms;
        }
        avg_sim = static_cast<float>(sum_sim / samples.size());
        avg_render = static_cast<float>(sum_render / samples.size());
    }
};

constexpr float kFixedDt = 1.0f / 240.0f;
constexpr int kMaxSubsteps = 16;
constexpr int kMouseButtonNone = 0;
constexpr int kMouseButtonDrag = 1;
constexpr int kMouseButtonPush = 2;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static const char* VS_SRC = R"(#version 450 core
layout(location = 0) in vec4 in_particle;
uniform float u_world_scale_inv;
uniform float u_point_size_px;
out float v_density;
void main() {
    gl_Position = vec4(in_particle.xy * u_world_scale_inv, 0.0, 1.0);
    v_density = in_particle.z;
    float density01 = clamp((in_particle.z - 0.85) / 0.35, 0.0, 1.0);
    gl_PointSize = u_point_size_px * mix(0.8, 1.4, density01);
}
)";

static const char* FS_SRC = R"(#version 450 core
in float v_density;
out vec4 frag;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    if (dot(c, c) > 1.0) discard;
    float density01 = clamp((v_density - 0.85) / 0.35, 0.0, 1.0);
    vec3 base = mix(vec3(0.20, 0.48, 0.82), vec3(0.72, 0.90, 1.00), density01);
    float edge = smoothstep(1.0, 0.25, dot(c, c));
    frag = vec4(base, edge);
}
)";

static const char* SURFACE_SPLAT_VS = R"(#version 450 core
layout(location = 0) in vec4 in_particle;
uniform float u_world_scale_inv;
uniform float u_splat_size_px;
void main() {
    gl_Position = vec4(in_particle.xy * u_world_scale_inv, 0.0, 1.0);
    gl_PointSize = u_splat_size_px;
}
)";

static const char* SURFACE_SPLAT_FS = R"(#version 450 core
out vec4 frag;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    float w = exp(-r2 * 3.0);
    frag = vec4(w, 0.0, 0.0, 0.0);
}
)";

static const char* FULLSCREEN_VS = R"(#version 450 core
out vec2 v_uv;
void main() {
    vec2 pos = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
)";

static const char* BACKGROUND_FS = R"(#version 450 core
in vec2 v_uv;
uniform float u_scale;
uniform float u_aspect;
uniform vec3 u_color0;
uniform vec3 u_color1;
out vec4 frag;

vec3 checker_color(vec2 uv, float scale, float aspect, vec3 c0, vec3 c1) {
    vec2 g = floor(vec2(uv.x * aspect, uv.y) * scale);
    float c = mod(g.x + g.y, 2.0);
    return mix(c0, c1, c);
}

void main() {
    frag = vec4(checker_color(v_uv, u_scale, u_aspect, u_color0, u_color1), 1.0);
}
)";

static const char* SURFACE_BLUR_FS = R"(#version 450 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec2 u_step;
out vec4 frag;
void main() {
    float w0 = 0.227027;
    float w1 = 0.194595;
    float w2 = 0.121622;
    float w3 = 0.054054;
    float w4 = 0.016216;
    float c = texture(u_tex, v_uv).r * w0;
    c += texture(u_tex, v_uv + u_step * 1.0).r * w1;
    c += texture(u_tex, v_uv - u_step * 1.0).r * w1;
    c += texture(u_tex, v_uv + u_step * 2.0).r * w2;
    c += texture(u_tex, v_uv - u_step * 2.0).r * w2;
    c += texture(u_tex, v_uv + u_step * 3.0).r * w3;
    c += texture(u_tex, v_uv - u_step * 3.0).r * w3;
    c += texture(u_tex, v_uv + u_step * 4.0).r * w4;
    c += texture(u_tex, v_uv - u_step * 4.0).r * w4;
    frag = vec4(c, 0.0, 0.0, 0.0);
}
)";

static const char* SURFACE_SHADE_FS = R"(#version 450 core
in vec2 v_uv;
uniform sampler2D u_density;
uniform float u_threshold;
uniform float u_smooth_width;
uniform vec3 u_base_color;
uniform vec3 u_highlight_color;
uniform float u_normal_strength;
uniform float u_rim_width_px;
uniform float u_refraction_strength;
uniform int   u_refraction_mode;
uniform float u_gradient_scale;
uniform float u_interior_blend_width;
uniform float u_interior_scale;
uniform float u_water_tint;
uniform float u_bg_scale;
uniform float u_bg_aspect;
uniform vec3 u_bg_color0;
uniform vec3 u_bg_color1;
out vec4 frag;

vec3 checker_color(vec2 uv, float scale, float aspect, vec3 c0, vec3 c1) {
    vec2 g = floor(vec2(uv.x * aspect, uv.y) * scale);
    float c = mod(g.x + g.y, 2.0);
    return mix(c0, c1, c);
}

void main() {
    float d = texture(u_density, v_uv).r;
    float a = smoothstep(u_threshold - u_smooth_width,
                         u_threshold + u_smooth_width, d);
    if (a <= 0.001) discard;
    vec2 ts = 1.0 / vec2(textureSize(u_density, 0));
    float dx = texture(u_density, v_uv + vec2(ts.x, 0.0)).r
             - texture(u_density, v_uv - vec2(ts.x, 0.0)).r;
    float dy = texture(u_density, v_uv + vec2(0.0, ts.y)).r
             - texture(u_density, v_uv - vec2(0.0, ts.y)).r;
    vec2 grad = vec2(dx, dy);
    float gmag = length(grad);
    vec2 dir = (gmag > 1.0e-6) ? grad / gmag : vec2(0.0);
    float iso_dist_px = (gmag > 1.0e-6)
        ? abs(d - u_threshold) / gmag
        : 1.0e6;
    float rim_mask = smoothstep(u_rim_width_px, 0.0, iso_dist_px);
    float tilt = clamp(u_normal_strength, 0.0, 0.95) * rim_mask;
    vec3 n = vec3(-dir * tilt, sqrt(max(1.0 - tilt * tilt, 0.0)));
    vec3 light = normalize(vec3(0.35, 0.55, 0.80));
    float diff = max(dot(n, light), 0.0);
    vec3 H = normalize(light + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, H), 0.0), 48.0);
    float fresnel = pow(1.0 - max(n.z, 0.0), 3.0);

    vec2 refract_uv;
    if (u_refraction_mode == 1) {
        // Option A: gradient-based — distorts wherever density varies
        refract_uv = v_uv - grad * u_refraction_strength * u_gradient_scale;
    } else if (u_refraction_mode == 2) {
        // Option B: gradient + density-proportional interior shift
        float depth = smoothstep(u_threshold, u_threshold + u_interior_blend_width, d);
        refract_uv = v_uv + n.xy * u_refraction_strength
                   - grad * u_refraction_strength * depth * u_interior_scale;
    } else {
        // Mode 0: original surface-only (rim-masked normal)
        refract_uv = v_uv + n.xy * u_refraction_strength;
    }
    vec3 bg = checker_color(refract_uv, u_bg_scale, u_bg_aspect,
                            u_bg_color0, u_bg_color1);
    vec3 tinted = mix(bg, u_base_color, clamp(u_water_tint, 0.0, 1.0));
    tinted *= 0.6 + 0.4 * diff;
    vec3 col = mix(tinted, u_highlight_color, fresnel * 0.5);
    col += vec3(1.0) * spec * 0.7;
    frag = vec4(col, a);
}
)";

static const char* OBSTACLE_VS = R"(#version 450 core
layout(location = 0) in vec2 in_pos;
uniform float u_world_scale_inv;
void main() {
    gl_Position = vec4(in_pos * u_world_scale_inv, 0.0, 1.0);
}
)";

static const char* OBSTACLE_FS = R"(#version 450 core
out vec4 frag;
void main() {
    frag = vec4(0.28, 0.30, 0.35, 0.92);
}
)";

static std::string get_scenes_dir() {
    char buf[4096] = {};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "scenes";
    return fs::path(std::string(buf, len)).parent_path() / "scenes";
}

static constexpr float kPiF = 3.14159265f;

static void append_circle_tris(std::vector<float2>& v,
                                float cx, float cy, float r) {
    constexpr int kSeg = 48;
    for (int i = 0; i < kSeg; ++i) {
        float a0 = 2.0f * kPiF * i       / kSeg;
        float a1 = 2.0f * kPiF * (i + 1) / kSeg;
        v.push_back({cx, cy});
        v.push_back({cx + r * cosf(a0), cy + r * sinf(a0)});
        v.push_back({cx + r * cosf(a1), cy + r * sinf(a1)});
    }
}

static void append_box_tris(std::vector<float2>& v,
                             float cx, float cy, float hw, float hh) {
    float x0 = cx - hw, x1 = cx + hw, y0 = cy - hh, y1 = cy + hh;
    v.push_back({x0, y0}); v.push_back({x1, y0}); v.push_back({x1, y1});
    v.push_back({x0, y0}); v.push_back({x1, y1}); v.push_back({x0, y1});
}

static std::vector<float2> build_obstacle_mesh(const SceneDesc& scene) {
    std::vector<float2> v;
    for (const Obstacle& obs : scene.obstacles) {
        if (obs.type == ObstacleType::Circle)
            append_circle_tris(v, obs.circle.cx, obs.circle.cy, obs.circle.r);
        else
            append_box_tris(v, obs.box.cx, obs.box.cy, obs.box.hw, obs.box.hh);
    }
    return v;
}

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

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "program link error: %s\n", log);
        std::exit(1);
    }
    return p;
}

struct SurfaceRenderer {
    GLuint splat_prog = 0;
    GLuint blur_prog = 0;
    GLuint shade_prog = 0;
    GLuint density_fbo = 0;
    GLuint density_tex = 0;
    GLuint blur_fbo[2] = {0, 0};
    GLuint blur_tex[2] = {0, 0};
    GLuint dummy_vao = 0;
    int fbo_w = 0;
    int fbo_h = 0;

    GLint splat_world_scale_loc = -1;
    GLint splat_size_loc = -1;
    GLint blur_step_loc = -1;
    GLint blur_tex_loc = -1;
    GLint shade_density_loc = -1;
    GLint shade_threshold_loc = -1;
    GLint shade_smooth_loc = -1;
    GLint shade_base_loc = -1;
    GLint shade_highlight_loc = -1;
    GLint shade_normal_strength_loc = -1;
    GLint shade_rim_width_loc = -1;
    GLint shade_refraction_strength_loc = -1;
    GLint shade_refraction_mode_loc = -1;
    GLint shade_gradient_scale_loc = -1;
    GLint shade_interior_blend_width_loc = -1;
    GLint shade_interior_scale_loc = -1;
    GLint shade_water_tint_loc = -1;
    GLint shade_bg_scale_loc = -1;
    GLint shade_bg_aspect_loc = -1;
    GLint shade_bg_color0_loc = -1;
    GLint shade_bg_color1_loc = -1;
};

struct SurfaceParams {
    bool enabled = false;
    float splat_size_world = 0.06f;
    float threshold = 0.55f;
    float smooth_width = 0.18f;
    int blur_iterations = 2;
    float normal_strength = 0.2f;
    float rim_width_px = 6.0f;
    float refraction_strength = 0.02f;
    int   refraction_mode = 0;
    float gradient_scale = 1.0f;
    float interior_blend_width = 0.4f;
    float interior_scale = 3.0f;
    float water_tint = 0.55f;
    float base_color[3] = {0.06f, 0.30f, 0.62f};
    float highlight_color[3] = {0.82f, 0.94f, 1.00f};
};

struct BackgroundParams {
    bool enabled = false;
    float scale = 16.0f;
    float color0[3] = {0.18f, 0.18f, 0.20f};
    float color1[3] = {0.30f, 0.30f, 0.34f};
    float solid_color[3] = {0.05f, 0.05f, 0.08f};
};

struct BackgroundRenderer {
    GLuint prog = 0;
    GLint scale_loc = -1;
    GLint aspect_loc = -1;
    GLint color0_loc = -1;
    GLint color1_loc = -1;
};

struct ObstacleRenderer {
    GLuint prog = 0;
    GLuint vao  = 0;
    GLuint vbo  = 0;
    int    vert_count = 0;
    GLint  world_scale_loc = -1;

    void init() {
        GLuint vs = compile_shader(GL_VERTEX_SHADER,   OBSTACLE_VS);
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, OBSTACLE_FS);
        prog = link_program(vs, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);
        world_scale_loc = glGetUniformLocation(prog, "u_world_scale_inv");
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), nullptr);
        glBindVertexArray(0);
    }

    void upload(const std::vector<float2>& verts) {
        vert_count = static_cast<int>(verts.size());
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vert_count * sizeof(float2),
                     verts.data(), GL_DYNAMIC_DRAW);
    }

    void draw(float world_scale_inv) const {
        if (vert_count == 0) return;
        glUseProgram(prog);
        glUniform1f(world_scale_loc, world_scale_inv);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, vert_count);
        glBindVertexArray(0);
    }

    void destroy() {
        if (vbo)  glDeleteBuffers(1, &vbo);
        if (vao)  glDeleteVertexArrays(1, &vao);
        if (prog) glDeleteProgram(prog);
    }
};

static void create_density_target(GLuint& fbo, GLuint& tex, int w, int h) {
    if (tex) glDeleteTextures(1, &tex);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "surface FBO incomplete\n");
        std::exit(1);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void surface_resize(SurfaceRenderer& sr, int w, int h) {
    if (w == sr.fbo_w && h == sr.fbo_h && sr.density_fbo != 0) return;
    create_density_target(sr.density_fbo, sr.density_tex, w, h);
    create_density_target(sr.blur_fbo[0], sr.blur_tex[0], w, h);
    create_density_target(sr.blur_fbo[1], sr.blur_tex[1], w, h);
    sr.fbo_w = w;
    sr.fbo_h = h;
}

static void surface_init(SurfaceRenderer& sr) {
    GLuint splat_vs = compile_shader(GL_VERTEX_SHADER, SURFACE_SPLAT_VS);
    GLuint splat_fs = compile_shader(GL_FRAGMENT_SHADER, SURFACE_SPLAT_FS);
    sr.splat_prog = link_program(splat_vs, splat_fs);
    glDeleteShader(splat_vs);
    glDeleteShader(splat_fs);

    GLuint quad_vs = compile_shader(GL_VERTEX_SHADER, FULLSCREEN_VS);
    GLuint blur_fs = compile_shader(GL_FRAGMENT_SHADER, SURFACE_BLUR_FS);
    sr.blur_prog = link_program(quad_vs, blur_fs);
    glDeleteShader(blur_fs);

    GLuint shade_fs = compile_shader(GL_FRAGMENT_SHADER, SURFACE_SHADE_FS);
    sr.shade_prog = link_program(quad_vs, shade_fs);
    glDeleteShader(shade_fs);
    glDeleteShader(quad_vs);

    sr.splat_world_scale_loc =
        glGetUniformLocation(sr.splat_prog, "u_world_scale_inv");
    sr.splat_size_loc = glGetUniformLocation(sr.splat_prog, "u_splat_size_px");
    sr.blur_step_loc = glGetUniformLocation(sr.blur_prog, "u_step");
    sr.blur_tex_loc = glGetUniformLocation(sr.blur_prog, "u_tex");
    sr.shade_density_loc = glGetUniformLocation(sr.shade_prog, "u_density");
    sr.shade_threshold_loc = glGetUniformLocation(sr.shade_prog, "u_threshold");
    sr.shade_smooth_loc = glGetUniformLocation(sr.shade_prog, "u_smooth_width");
    sr.shade_base_loc = glGetUniformLocation(sr.shade_prog, "u_base_color");
    sr.shade_highlight_loc =
        glGetUniformLocation(sr.shade_prog, "u_highlight_color");
    sr.shade_normal_strength_loc =
        glGetUniformLocation(sr.shade_prog, "u_normal_strength");
    sr.shade_rim_width_loc =
        glGetUniformLocation(sr.shade_prog, "u_rim_width_px");
    sr.shade_refraction_strength_loc =
        glGetUniformLocation(sr.shade_prog, "u_refraction_strength");
    sr.shade_refraction_mode_loc =
        glGetUniformLocation(sr.shade_prog, "u_refraction_mode");
    sr.shade_gradient_scale_loc =
        glGetUniformLocation(sr.shade_prog, "u_gradient_scale");
    sr.shade_interior_blend_width_loc =
        glGetUniformLocation(sr.shade_prog, "u_interior_blend_width");
    sr.shade_interior_scale_loc =
        glGetUniformLocation(sr.shade_prog, "u_interior_scale");
    sr.shade_water_tint_loc =
        glGetUniformLocation(sr.shade_prog, "u_water_tint");
    sr.shade_bg_scale_loc =
        glGetUniformLocation(sr.shade_prog, "u_bg_scale");
    sr.shade_bg_aspect_loc =
        glGetUniformLocation(sr.shade_prog, "u_bg_aspect");
    sr.shade_bg_color0_loc =
        glGetUniformLocation(sr.shade_prog, "u_bg_color0");
    sr.shade_bg_color1_loc =
        glGetUniformLocation(sr.shade_prog, "u_bg_color1");

    glGenVertexArrays(1, &sr.dummy_vao);
}

static void background_init(BackgroundRenderer& br) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, FULLSCREEN_VS);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, BACKGROUND_FS);
    br.prog = link_program(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    br.scale_loc = glGetUniformLocation(br.prog, "u_scale");
    br.aspect_loc = glGetUniformLocation(br.prog, "u_aspect");
    br.color0_loc = glGetUniformLocation(br.prog, "u_color0");
    br.color1_loc = glGetUniformLocation(br.prog, "u_color1");
}

static void background_destroy(BackgroundRenderer& br) {
    if (br.prog) glDeleteProgram(br.prog);
    br.prog = 0;
}

static void surface_destroy(SurfaceRenderer& sr) {
    if (sr.density_fbo) glDeleteFramebuffers(1, &sr.density_fbo);
    if (sr.density_tex) glDeleteTextures(1, &sr.density_tex);
    for (int i = 0; i < 2; ++i) {
        if (sr.blur_fbo[i]) glDeleteFramebuffers(1, &sr.blur_fbo[i]);
        if (sr.blur_tex[i]) glDeleteTextures(1, &sr.blur_tex[i]);
    }
    if (sr.dummy_vao) glDeleteVertexArrays(1, &sr.dummy_vao);
    if (sr.splat_prog) glDeleteProgram(sr.splat_prog);
    if (sr.blur_prog) glDeleteProgram(sr.blur_prog);
    if (sr.shade_prog) glDeleteProgram(sr.shade_prog);
}

static void surface_render(SurfaceRenderer& sr, const SurfaceParams& params,
                           const BackgroundParams& bg_params,
                           GLuint particle_vao, int particle_count, int w,
                           int h, float world_scale_inv) {
    surface_resize(sr, w, h);

    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, sr.density_fbo);
    glViewport(0, 0, w, h);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);

    float splat_size_px =
        params.splat_size_world * world_scale_inv * static_cast<float>(h);
    if (splat_size_px < 2.0f) splat_size_px = 2.0f;

    glUseProgram(sr.splat_prog);
    glUniform1f(sr.splat_world_scale_loc, world_scale_inv);
    glUniform1f(sr.splat_size_loc, splat_size_px);
    glBindVertexArray(particle_vao);
    glDrawArrays(GL_POINTS, 0, particle_count);
    glBindVertexArray(0);

    glDisable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint src_tex = sr.density_tex;
    int read_idx = 0;
    glUseProgram(sr.blur_prog);
    glUniform1i(sr.blur_tex_loc, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(sr.dummy_vao);

    int iters = params.blur_iterations;
    if (iters < 0) iters = 0;
    if (iters > 8) iters = 8;
    for (int i = 0; i < iters; ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, sr.blur_fbo[read_idx]);
        glViewport(0, 0, w, h);
        glBindTexture(GL_TEXTURE_2D, src_tex);
        glUniform2f(sr.blur_step_loc, 1.0f / static_cast<float>(w), 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        int write_idx = 1 - read_idx;
        glBindFramebuffer(GL_FRAMEBUFFER, sr.blur_fbo[write_idx]);
        glBindTexture(GL_TEXTURE_2D, sr.blur_tex[read_idx]);
        glUniform2f(sr.blur_step_loc, 0.0f, 1.0f / static_cast<float>(h));
        glDrawArrays(GL_TRIANGLES, 0, 3);

        src_tex = sr.blur_tex[write_idx];
        read_idx = write_idx;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, w, h);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(sr.shade_prog);
    glUniform1i(sr.shade_density_loc, 0);
    glUniform1f(sr.shade_threshold_loc, params.threshold);
    glUniform1f(sr.shade_smooth_loc, params.smooth_width);
    glUniform1f(sr.shade_normal_strength_loc, params.normal_strength);
    glUniform1f(sr.shade_rim_width_loc, params.rim_width_px);
    glUniform3fv(sr.shade_base_loc, 1, params.base_color);
    glUniform3fv(sr.shade_highlight_loc, 1, params.highlight_color);
    glUniform1f(sr.shade_refraction_strength_loc, params.refraction_strength);
    glUniform1i(sr.shade_refraction_mode_loc, params.refraction_mode);
    glUniform1f(sr.shade_gradient_scale_loc, params.gradient_scale);
    glUniform1f(sr.shade_interior_blend_width_loc, params.interior_blend_width);
    glUniform1f(sr.shade_interior_scale_loc, params.interior_scale);
    glUniform1f(sr.shade_water_tint_loc, params.water_tint);
    glUniform1f(sr.shade_bg_scale_loc, bg_params.scale);
    float aspect = (h > 0) ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;
    glUniform1f(sr.shade_bg_aspect_loc, aspect);
    if (bg_params.enabled) {
        glUniform3fv(sr.shade_bg_color0_loc, 1, bg_params.color0);
        glUniform3fv(sr.shade_bg_color1_loc, 1, bg_params.color1);
    } else {
        glUniform3fv(sr.shade_bg_color0_loc, 1, bg_params.solid_color);
        glUniform3fv(sr.shade_bg_color1_loc, 1, bg_params.solid_color);
    }
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, src_tex);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

struct MouseController {
    double cursor_x = 0.0;
    double cursor_y = 0.0;
    bool has_cursor = false;
    bool left_down = false;
    bool right_down = false;
    float radius = 0.14f;
    float strength = 16.0f;
    float2 prev_world = make_float2(0.0f, 0.0f);
    bool has_prev_world = false;
};

static void cursor_position_callback(GLFWwindow* window, double x, double y) {
    auto* mouse = static_cast<MouseController*>(glfwGetWindowUserPointer(window));
    if (!mouse) {
        return;
    }
    mouse->cursor_x = x;
    mouse->cursor_y = y;
    mouse->has_cursor = true;
}

static void mouse_button_callback(GLFWwindow* window, int button, int action,
                                  int /*mods*/) {
    auto* mouse = static_cast<MouseController*>(glfwGetWindowUserPointer(window));
    if (!mouse) {
        return;
    }

    bool down = action != GLFW_RELEASE;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouse->left_down = down;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse->right_down = down;
    }
}

static float2 screen_to_world(double x, double y, int width, int height) {
    if (width <= 0 || height <= 0) {
        return make_float2(0.0f, 0.0f);
    }

    float ndc_x = static_cast<float>((x / static_cast<double>(width)) * 2.0 - 1.0);
    float ndc_y =
        static_cast<float>(1.0 - (y / static_cast<double>(height)) * 2.0);
    return make_float2(ndc_x * kWorldHalfExtent, ndc_y * kWorldHalfExtent);
}

static MouseState build_mouse_state(MouseController& mouse, int width, int height,
                                    float frame_dt, bool capture_mouse) {
    MouseState state{};
    state.radius = mouse.radius;
    state.strength = mouse.strength;
    state.button_state = kMouseButtonNone;

    if (!mouse.has_cursor) {
        mouse.has_prev_world = false;
        return state;
    }

    float2 world = screen_to_world(mouse.cursor_x, mouse.cursor_y, width, height);
    float2 world_vel = make_float2(0.0f, 0.0f);
    if (mouse.has_prev_world && frame_dt > 1.0e-6f) {
        world_vel = make_float2((world.x - mouse.prev_world.x) / frame_dt,
                                (world.y - mouse.prev_world.y) / frame_dt);
    }

    mouse.prev_world = world;
    mouse.has_prev_world = true;
    state.pos = world;
    state.vel = world_vel;

    if (!capture_mouse) {
        if (mouse.right_down) {
            state.button_state = kMouseButtonPush;
        } else if (mouse.left_down) {
            state.button_state = kMouseButtonDrag;
        }
    }

    return state;
}

int main() {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* win = glfwCreateWindow(1024, 1024, "water-sim2 — Phase 5", nullptr, nullptr);
    if (!win) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    MouseController mouse{};
    glfwSetWindowUserPointer(win, &mouse);
    glfwSetCursorPosCallback(win, cursor_position_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "glewInit failed\n");
        return 1;
    }
    while (glGetError() != GL_NO_ERROR) {}

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("CUDA device: %s (cc %d.%d), %zu MB\n", prop.name, prop.major,
                prop.minor, prop.totalGlobalMem / (1024 * 1024));
    CUDA_CHECK(cudaSetDevice(dev));
    init_simulation();
    const int particle_count = get_particle_count();

    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, particle_count * sizeof(float) * 4, nullptr,
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, nullptr);
    glBindVertexArray(0);

    cudaGraphicsResource* cuda_vbo = nullptr;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, cudaGraphicsRegisterFlagsNone));

    GLuint vs = compile_shader(GL_VERTEX_SHADER, VS_SRC);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, FS_SRC);
    GLuint prog = link_program(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint u_world_scale_inv_loc = glGetUniformLocation(prog, "u_world_scale_inv");
    GLint u_point_size_px_loc = glGetUniformLocation(prog, "u_point_size_px");
    constexpr float kParticleRadiusWorld = 0.0075f;

    SurfaceRenderer surface{};
    surface_init(surface);
    SurfaceParams surface_params{};

    BackgroundRenderer background{};
    background_init(background);
    BackgroundParams bg_params{};

    ObstacleRenderer obstacle_renderer{};
    obstacle_renderer.init();

    // Scene state — all scenes driven by JSON files
    SceneDesc current_scene{};
    std::vector<std::string> scene_files;
    std::vector<std::string> scene_names;   // display names parallel to scene_files
    int active_scene_idx = 0;
    std::string scene_load_error;

    scene_files = list_scene_files(get_scenes_dir());
    for (const auto& f : scene_files)
        scene_names.push_back(fs::path(f).stem().string());

    // Helper: load a scene by index into the current state
    auto load_scene = [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(scene_files.size())) return;
        SceneDesc desc;
        std::string err;
        if (!load_scene_json(scene_files[idx], desc, err)) {
            scene_load_error = err;
            return;
        }
        std::vector<float> sdf_px;
        bake_sdf(desc, kWorldHalfExtent, kSdfResolution, sdf_px);
        if (desc.obstacles.empty())
            clear_sdf();
        else
            upload_sdf(sdf_px.data(), kSdfResolution);
        std::vector<float2> init_pos;
        seed_from_scene_desc(desc, get_particle_count(), init_pos);
        set_initial_positions(init_pos);
        current_scene = std::move(desc);
        obstacle_renderer.upload(build_obstacle_mesh(current_scene));
        reset_simulation();
        scene_load_error.clear();
    };

    // Auto-load first scene on startup
    if (!scene_files.empty()) {
        load_scene(0);
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 450 core");

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    double last_time = glfwGetTime();
    double accumulator = kFixedDt;
    SimulationStats stats{};
    FrameTimingWindow frame_window{};
    float last_render_ms = 0.0f;
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        double now = glfwGetTime();
        double frame_dt = now - last_time;
        last_time = now;
        if (frame_dt > 0.05) {
            frame_dt = 0.05;
        }
        accumulator += frame_dt;

        int w = 0, h = 0;
        glfwGetFramebufferSize(win, &w, &h);
        MouseState sim_mouse =
            build_mouse_state(mouse, w, h, static_cast<float>(frame_dt),
                              ImGui::GetIO().WantCaptureMouse);

        double sim_start = glfwGetTime();
        float4* dptr = nullptr;
        size_t bytes = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dptr), &bytes, cuda_vbo));
        int substeps = 0;
        while (accumulator >= kFixedDt && substeps < kMaxSubsteps) {
            step_simulation(kFixedDt, sim_mouse, dptr);
            accumulator -= kFixedDt;
            ++substeps;
        }
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo, 0));
        stats = get_simulation_stats();
        double sim_end = glfwGetTime();
        float sim_ms = static_cast<float>((sim_end - sim_start) * 1000.0);
        frame_window.push(sim_end, sim_ms, last_render_ms);

        double render_start = glfwGetTime();
        float world_scale_inv = 1.0f / kWorldHalfExtent;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, w, h);
        if (bg_params.enabled) {
            glDisable(GL_BLEND);
            glUseProgram(background.prog);
            glUniform1f(background.scale_loc, bg_params.scale);
            float bg_aspect =
                (h > 0) ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;
            glUniform1f(background.aspect_loc, bg_aspect);
            glUniform3fv(background.color0_loc, 1, bg_params.color0);
            glUniform3fv(background.color1_loc, 1, bg_params.color1);
            glBindVertexArray(surface.dummy_vao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindVertexArray(0);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        } else {
            glClearColor(bg_params.solid_color[0], bg_params.solid_color[1],
                         bg_params.solid_color[2], 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
        }

        if (surface_params.enabled) {
            surface_render(surface, surface_params, bg_params, vao,
                           particle_count, w, h, world_scale_inv);
        } else {
            glUseProgram(prog);
            float point_size_px =
                kParticleRadiusWorld * world_scale_inv * static_cast<float>(h);
            glUniform1f(u_world_scale_inv_loc, world_scale_inv);
            glUniform1f(u_point_size_px_loc, point_size_px);
            glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, particle_count);
            glBindVertexArray(0);
        }

        // Obstacles rendered on top (solid, opaque)
        glDisable(GL_BLEND);
        obstacle_renderer.draw(world_scale_inv);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
            ImGui::Begin("stats", nullptr,
                         ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoBackground |
                             ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("CUDA: %s", prop.name);
            ImGui::Text("Particles: %d", particle_count);
            ImGui::Text("Solver iterations: %d", get_solver_iterations());
            float avg_sim = 0.0f, max_sim = 0.0f;
            float avg_render = 0.0f, max_render = 0.0f;
            frame_window.summarize(avg_sim, max_sim, avg_render, max_render);
            ImGui::Text("Sim    avg/max: %.2f / %.2f ms", avg_sim, max_sim);
            ImGui::Text("Render avg/max: %.2f / %.2f ms", avg_render, max_render);
            ImGui::Text("Avg density: %.2f", stats.avg_density);
            ImGui::Text("Max density: %.2f", stats.max_density);
            ImGui::Text("Avg speed: %.3f", stats.avg_speed);
            ImGui::Separator();
            ImGui::Text("Left drag: pull fluid");
            ImGui::Text("Right drag: push fluid");
            ImGui::SliderFloat("Mouse radius", &mouse.radius, 0.05f, 1.0f);
            ImGui::SliderFloat("Mouse strength", &mouse.strength, 2.0f, 200.0f);

            if (!scene_files.empty()) {
                // Build parallel c-string array for the combo
                std::vector<const char*> name_ptrs;
                name_ptrs.reserve(scene_names.size());
                for (const auto& n : scene_names) name_ptrs.push_back(n.c_str());

                ImGui::PushItemWidth(170.0f);
                if (ImGui::Combo("scene", &active_scene_idx,
                                 name_ptrs.data(),
                                 static_cast<int>(name_ptrs.size()))) {
                    load_scene(active_scene_idx);
                    accumulator = kFixedDt;
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::TextDisabled("(no scenes/*.json found)");
            }
            if (!scene_load_error.empty())
                ImGui::TextColored({1.0f, 0.3f, 0.3f, 1.0f}, "%s",
                                   scene_load_error.c_str());
            if (ImGui::Button("Reset")) {
                load_scene(active_scene_idx);
                accumulator = kFixedDt;
            }
            ImGui::End();
        }

        {
            ImGui::SetNextWindowPos(ImVec2(static_cast<float>(w) - 10.0f, 10.0f),
                                    ImGuiCond_FirstUseEver, ImVec2(1.0f, 0.0f));
            ImGui::SetNextWindowSize(ImVec2(280.0f, 0.0f), ImGuiCond_FirstUseEver);
            ImGui::Begin("parameters");

            TunableParams tp = get_tunable_params();
            bool changed = false;
            ImGui::PushItemWidth(120.0f);
            changed |= ImGui::InputFloat("rest density", &tp.rest_density, 0.0f, 0.0f,
                                         "%.2f");
            changed |= ImGui::InputFloat("kernel radius", &tp.kernel_radius, 0.0f, 0.0f,
                                         "%.4f");
            changed |= ImGui::InputInt("solver iterations", &tp.solver_iterations, 0, 0);
            changed |= ImGui::InputFloat("lambda_epsilon", &tp.lambda_epsilon, 0.0f, 0.0f,
                                         "%.4f");
            changed |= ImGui::InputFloat("tensile_k", &tp.tensile_k, 0.0f, 0.0f,
                                         "%.6f");
            changed |= ImGui::InputFloat("tensile_n", &tp.tensile_n, 0.0f, 0.0f,
                                         "%.3f");
            changed |= ImGui::InputFloat("tensile_q", &tp.tensile_q, 0.0f, 0.0f,
                                         "%.3f");
            changed |= ImGui::InputFloat("gravity", &tp.gravity, 0.0f, 0.0f,
                                         "%.3f");
            changed |= ImGui::InputFloat("velocity damping", &tp.velocity_damping,
                                         0.0f, 0.0f, "%.4f");
            changed |= ImGui::InputFloat("boundary bounce", &tp.boundary_bounce,
                                         0.0f, 0.0f, "%.3f");
            changed |= ImGui::InputFloat("viscosity_c (XSPH)", &tp.viscosity_c,
                                         0.0f, 0.0f, "%.6f");
            changed |= ImGui::InputFloat("vorticity_eps", &tp.vorticity_eps,
                                         0.0f, 0.0f, "%.6f");
            changed |= ImGui::InputFloat("max speed", &tp.max_speed, 0.0f, 0.0f,
                                         "%.3f");
            changed |= ImGui::InputFloat("max pos correction",
                                         &tp.max_position_correction, 0.0f,
                                         0.0f, "%.5f");
            ImGui::PopItemWidth();
            if (changed) {
                set_tunable_params(tp);
            }

            ImGui::Separator();
            ImGui::Text("surface render");
            ImGui::Checkbox("enabled##surface", &surface_params.enabled);
            ImGui::PushItemWidth(120.0f);
            ImGui::InputFloat("splat size (world)", &surface_params.splat_size_world,
                              0.0f, 0.0f, "%.4f");
            ImGui::InputFloat("threshold", &surface_params.threshold, 0.0f, 0.0f,
                              "%.3f");
            ImGui::InputFloat("smooth width", &surface_params.smooth_width, 0.0f,
                              0.0f, "%.3f");
            ImGui::InputInt("blur iterations", &surface_params.blur_iterations, 0, 0);
            ImGui::InputFloat("normal strength", &surface_params.normal_strength,
                              0.0f, 0.0f, "%.2f");
            ImGui::InputFloat("rim width (px)", &surface_params.rim_width_px,
                              0.0f, 0.0f, "%.2f");
            ImGui::SliderFloat("refraction strength",
                               &surface_params.refraction_strength, 0.0f, 0.1f,
                               "%.4f");
            ImGui::PushItemWidth(170.0f);
            ImGui::Combo("refraction mode", &surface_params.refraction_mode,
                         "surface only\0gradient\0gradient+depth\0");
            ImGui::PopItemWidth();
            if (surface_params.refraction_mode == 1) {
                ImGui::SliderFloat("gradient scale",
                                   &surface_params.gradient_scale, 0.0f, 10.0f,
                                   "%.2f");
            }
            if (surface_params.refraction_mode == 2) {
                ImGui::SliderFloat("interior blend width",
                                   &surface_params.interior_blend_width, 0.01f,
                                   2.0f, "%.3f");
                ImGui::SliderFloat("interior scale",
                                   &surface_params.interior_scale, 0.0f, 10.0f,
                                   "%.2f");
            }
            ImGui::SliderFloat("water tint", &surface_params.water_tint, 0.0f,
                               1.0f, "%.2f");
            ImGui::PopItemWidth();
            ImGui::ColorEdit3("base", surface_params.base_color);
            ImGui::ColorEdit3("highlight", surface_params.highlight_color);

            ImGui::Separator();
            ImGui::Text("background");
            ImGui::Checkbox("checker bg", &bg_params.enabled);
            ImGui::BeginDisabled(!bg_params.enabled);
            ImGui::PushItemWidth(120.0f);
            ImGui::SliderFloat("bg scale", &bg_params.scale, 1.0f, 64.0f, "%.1f");
            ImGui::PopItemWidth();
            ImGui::ColorEdit3("bg color 0", bg_params.color0);
            ImGui::ColorEdit3("bg color 1", bg_params.color1);
            ImGui::EndDisabled();
            ImGui::ColorEdit3("bg solid", bg_params.solid_color);
            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        double render_end = glfwGetTime();
        last_render_ms = static_cast<float>((render_end - render_start) * 1000.0);

        glfwSwapBuffers(win);
    }

    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo));
    shutdown_simulation();
    obstacle_renderer.destroy();
    surface_destroy(surface);
    background_destroy(background);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
