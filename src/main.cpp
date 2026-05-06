#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <unistd.h>

#include "kernel.cuh"

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct FrameSample { double t; float sim_ms; float render_ms; };

struct FrameTimingWindow {
    static constexpr double kWindowSeconds = 1.0;
    std::deque<FrameSample> samples;

    void push(double now, float sim_ms, float render_ms) {
        samples.push_back({now, sim_ms, render_ms});
        while (!samples.empty() && now - samples.front().t > kWindowSeconds)
            samples.pop_front();
    }

    void summarize(float& as, float& ms, float& ar, float& mr) const {
        as = ms = ar = mr = 0.0f;
        if (samples.empty()) return;
        double ss = 0.0, sr = 0.0;
        for (const auto& s : samples) {
            ss += s.sim_ms; sr += s.render_ms;
            if (s.sim_ms > ms)    ms = s.sim_ms;
            if (s.render_ms > mr) mr = s.render_ms;
        }
        int n = (int)samples.size();
        as = (float)(ss / n);
        ar = (float)(sr / n);
    }
};

constexpr float kFixedDt     = 1.0f / 240.0f;
constexpr int   kMaxSubsteps = 16;
constexpr int   kMouseButtonNone = 0;
constexpr int   kMouseButtonPush = 2;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Shaders — 3-D impostor sphere particles
// ---------------------------------------------------------------------------

static const char* PARTICLE_VS = R"(#version 450 core
layout(location = 0) in vec4 in_particle; // xyz=pos, w=density01

uniform mat4  u_mv;
uniform mat4  u_mvp;
uniform float u_radius_world;
uniform float u_proj11;       // projection[1][1] = cot(fovy/2)
uniform float u_viewport_h;

out float v_density;
out float v_eye_z;

void main() {
    vec4 pos_eye   = u_mv * vec4(in_particle.xyz, 1.0);
    v_eye_z        = pos_eye.z;
    v_density      = in_particle.w;
    float eye_neg  = -pos_eye.z;
    gl_PointSize   = max(1.0, u_radius_world * u_proj11 * u_viewport_h / eye_neg);
    gl_Position    = u_mvp * vec4(in_particle.xyz, 1.0);
}
)";

static const char* PARTICLE_FS = R"(#version 450 core
in float v_density;
in float v_eye_z;

uniform float u_sphere_radius;
uniform float u_proj22;   // proj[2][2]
uniform float u_proj32;   // proj[3][2]

out vec4 frag;

void main() {
    vec2  uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float sphere_z = sqrt(1.0 - r2);
    vec3  n        = vec3(uv.x, -uv.y, sphere_z);   // view-space sphere normal

    // Correct fragment depth to sphere surface
    float eye_z  = v_eye_z + u_sphere_radius * sphere_z;
    float ndc_z  = (u_proj22 * eye_z + u_proj32) / (-eye_z);
    gl_FragDepth = ndc_z * 0.5 + 0.5;

    // Phong
    vec3  light = normalize(vec3(0.45, 0.75, 0.55));
    float diff  = max(dot(n, light), 0.0);
    vec3  H     = normalize(light + vec3(0.0, 0.0, 1.0));
    float spec  = pow(max(dot(n, H), 0.0), 64.0);
    float fres  = pow(1.0 - sphere_z, 3.0);

    float d    = clamp((v_density - 0.8) / 0.4, 0.0, 1.0);
    vec3  base = mix(vec3(0.06, 0.28, 0.65), vec3(0.50, 0.82, 1.00), d);
    vec3  col  = base * (0.15 + 0.85 * diff);
    col = mix(col, vec3(0.85, 0.95, 1.00), fres * 0.4);
    col += vec3(spec * 0.55);
    frag = vec4(col, 1.0);
}
)";

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

struct Camera {
    float     yaw      = 0.5f;
    float     pitch    = 0.3f;
    float     dist     = 10.0f;
    glm::vec3 target   = {0.0f, 0.0f, 0.0f};
    float     fovy_deg = 45.0f;
    float     near_z   = 0.1f;
    float     far_z    = 200.0f;

    glm::vec3 position() const {
        return target + glm::vec3(
            cosf(pitch) * sinf(yaw),
            sinf(pitch),
            cosf(pitch) * cosf(yaw)) * dist;
    }

    glm::mat4 view() const {
        return glm::lookAt(position(), target, glm::vec3(0, 1, 0));
    }

    glm::mat4 proj(float aspect) const {
        return glm::perspective(glm::radians(fovy_deg), aspect, near_z, far_z);
    }
};

// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------

struct InputState {
    double cursor_x       = 0.0;
    double cursor_y       = 0.0;
    double prev_cursor_x  = 0.0;
    double prev_cursor_y  = 0.0;
    bool   has_cursor     = false;
    bool   left_down      = false;
    bool   right_down     = false;
    float  scroll_dy      = 0.0f;
    float  mouse_radius   = 0.5f;
    float  mouse_strength = 50.0f;
};

static void cursor_position_callback(GLFWwindow* win, double x, double y) {
    auto* s = static_cast<InputState*>(glfwGetWindowUserPointer(win));
    if (!s) return;
    s->cursor_x   = x;
    s->cursor_y   = y;
    s->has_cursor = true;
}

static void mouse_button_callback(GLFWwindow* win, int button, int action, int) {
    auto* s = static_cast<InputState*>(glfwGetWindowUserPointer(win));
    if (!s) return;
    bool down = (action != GLFW_RELEASE);
    if (button == GLFW_MOUSE_BUTTON_LEFT)  s->left_down  = down;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) s->right_down = down;
}

static void scroll_callback(GLFWwindow* win, double, double yoff) {
    auto* s = static_cast<InputState*>(glfwGetWindowUserPointer(win));
    if (s) s->scroll_dy += static_cast<float>(yoff);
}

// ---------------------------------------------------------------------------
// Per-frame helpers
// ---------------------------------------------------------------------------

static void update_camera(Camera& cam, InputState& inp, bool capture) {
    if (inp.scroll_dy != 0.0f) {
        cam.dist *= powf(0.88f, inp.scroll_dy);
        cam.dist  = glm::clamp(cam.dist, 1.0f, 80.0f);
        inp.scroll_dy = 0.0f;
    }
    if (!capture && inp.has_cursor && inp.left_down) {
        float dx = static_cast<float>(inp.cursor_x - inp.prev_cursor_x) * 0.005f;
        float dy = static_cast<float>(inp.cursor_y - inp.prev_cursor_y) * 0.005f;
        cam.yaw   -= dx;
        cam.pitch  = glm::clamp(cam.pitch + dy, -1.50f, 1.50f);
    }
    inp.prev_cursor_x = inp.cursor_x;
    inp.prev_cursor_y = inp.cursor_y;
}

static MouseState build_mouse_state(const InputState& inp, const Camera& cam,
                                     int w, int h, bool capture) {
    MouseState ms{};
    ms.radius       = inp.mouse_radius;
    ms.strength     = inp.mouse_strength;
    ms.button_state = kMouseButtonNone;

    if (capture || !inp.has_cursor || !inp.right_down) return ms;

    float aspect = (h > 0) ? static_cast<float>(w) / h : 1.0f;
    glm::mat4 inv_view = glm::inverse(cam.view());
    glm::mat4 inv_proj = glm::inverse(cam.proj(aspect));

    float ndc_x =  2.0f * static_cast<float>(inp.cursor_x) / w - 1.0f;
    float ndc_y = -(2.0f * static_cast<float>(inp.cursor_y) / h - 1.0f);

    glm::vec4 ray_eye = inv_proj * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);
    glm::vec3 dir = glm::normalize(glm::vec3(inv_view * ray_eye));
    glm::vec3 ori = cam.position();

    ms.ray_origin   = make_float3(ori.x, ori.y, ori.z);
    ms.ray_dir      = make_float3(dir.x, dir.y, dir.z);
    ms.button_state = kMouseButtonPush;
    return ms;
}

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

static GLuint link_program(GLuint vs, GLuint fs) {
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
    return p;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);

    GLFWwindow* win = glfwCreateWindow(1280, 800, "water-sim2 — Phase 6", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "glfwCreateWindow failed\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    InputState inp{};
    glfwSetWindowUserPointer(win, &inp);
    glfwSetCursorPosCallback(win,  cursor_position_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetScrollCallback(win,      scroll_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::fprintf(stderr, "glewInit failed\n"); return 1; }
    while (glGetError() != GL_NO_ERROR) {}

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("CUDA device: %s (cc %d.%d), %zu MB\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / (1024 * 1024));
    CUDA_CHECK(cudaSetDevice(dev));
    init_simulation();
    const int particle_count = get_particle_count();

    // --- CUDA/GL interop VBO -------------------------------------------------
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, particle_count * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, nullptr);
    glBindVertexArray(0);

    cudaGraphicsResource* cuda_vbo = nullptr;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, cudaGraphicsRegisterFlagsNone));

    // --- Particle shader -----------------------------------------------------
    GLuint vs   = compile_shader(GL_VERTEX_SHADER,   PARTICLE_VS);
    GLuint fs   = compile_shader(GL_FRAGMENT_SHADER, PARTICLE_FS);
    GLuint prog = link_program(vs, fs);
    glDeleteShader(vs); glDeleteShader(fs);

    GLint loc_mv     = glGetUniformLocation(prog, "u_mv");
    GLint loc_mvp    = glGetUniformLocation(prog, "u_mvp");
    GLint loc_rad    = glGetUniformLocation(prog, "u_radius_world");
    GLint loc_p11    = glGetUniformLocation(prog, "u_proj11");
    GLint loc_vph    = glGetUniformLocation(prog, "u_viewport_h");
    GLint loc_srad   = glGetUniformLocation(prog, "u_sphere_radius");
    GLint loc_p22    = glGetUniformLocation(prog, "u_proj22");
    GLint loc_p32    = glGetUniformLocation(prog, "u_proj32");

    constexpr float kParticleRadius = 0.05f;

    // --- State ---------------------------------------------------------------
    Camera cam{};
    SceneId active_scene = SceneId::CubeFull;
    float bg_color[3]    = {0.04f, 0.05f, 0.08f};

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 450 core");

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

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

        double now      = glfwGetTime();
        double frame_dt = now - last_time;
        last_time = now;
        if (frame_dt > 0.05) frame_dt = 0.05;
        accumulator += frame_dt;

        int w = 0, h = 0;
        glfwGetFramebufferSize(win, &w, &h);
        bool capture = ImGui::GetIO().WantCaptureMouse;

        update_camera(cam, inp, capture);
        MouseState sim_mouse = build_mouse_state(inp, cam, w, h, capture);

        // --- Simulate --------------------------------------------------------
        double sim_start = glfwGetTime();
        float4* dptr = nullptr; size_t bytes = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&dptr), &bytes, cuda_vbo));
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

        // --- Render ----------------------------------------------------------
        double render_start = glfwGetTime();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, w, h);
        glClearColor(bg_color[0], bg_color[1], bg_color[2], 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (h > 0) ? static_cast<float>(w) / h : 1.0f;
        glm::mat4 proj_mat = cam.proj(aspect);
        glm::mat4 view_mat = cam.view();
        glm::mat4 mvp      = proj_mat * view_mat;

        glDisable(GL_BLEND);
        glUseProgram(prog);
        glUniformMatrix4fv(loc_mv,  1, GL_FALSE, glm::value_ptr(view_mat));
        glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1f(loc_rad,  kParticleRadius);
        glUniform1f(loc_p11,  proj_mat[1][1]);
        glUniform1f(loc_vph,  static_cast<float>(h));
        glUniform1f(loc_srad, kParticleRadius);
        glUniform1f(loc_p22,  proj_mat[2][2]);
        glUniform1f(loc_p32,  proj_mat[3][2]);

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, particle_count);
        glBindVertexArray(0);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // --- ImGui -----------------------------------------------------------
        {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
            ImGui::Begin("stats", nullptr,
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("CUDA: %s", prop.name);
            ImGui::Text("Particles: %d", particle_count);
            ImGui::Text("Solver iterations: %d", get_solver_iterations());
            float as = 0, ms = 0, ar = 0, mr = 0;
            frame_window.summarize(as, ms, ar, mr);
            ImGui::Text("Sim    avg/max: %.2f / %.2f ms", as, ms);
            ImGui::Text("Render avg/max: %.2f / %.2f ms", ar, mr);
            ImGui::Text("Avg density: %.2f", stats.avg_density);
            ImGui::Text("Max density: %.2f", stats.max_density);
            ImGui::Text("Avg speed: %.3f",   stats.avg_speed);
            ImGui::Text("Grid Y: %d / %d cells", stats.grid_h, stats.grid_h_max);
            ImGui::Separator();
            ImGui::Text("Left drag: orbit  |  Scroll: zoom");
            ImGui::Text("Right drag: push fluid");
            ImGui::SliderFloat("Mouse radius",   &inp.mouse_radius,   0.1f, 2.0f);
            ImGui::SliderFloat("Mouse strength", &inp.mouse_strength, 5.0f, 500.0f);

            const char* scene_names[] = { "cube (full)", "column (left)", "wide block" };
            int scene_idx = static_cast<int>(active_scene);
            ImGui::PushItemWidth(150.0f);
            if (ImGui::Combo("scene", &scene_idx, scene_names, kSceneCount)) {
                active_scene = static_cast<SceneId>(scene_idx);
                set_active_scene(active_scene);
                accumulator = kFixedDt;
            }
            ImGui::PopItemWidth();
            if (ImGui::Button("Reset")) {
                set_active_scene(active_scene);
                accumulator = kFixedDt;
            }
            ImGui::ColorEdit3("bg color", bg_color);
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
            changed |= ImGui::InputFloat("rest density",  &tp.rest_density,  0, 0, "%.2f");
            changed |= ImGui::InputFloat("kernel radius", &tp.kernel_radius, 0, 0, "%.4f");
            changed |= ImGui::InputInt("solver iterations", &tp.solver_iterations, 0, 0);
            changed |= ImGui::InputFloat("lambda_epsilon",  &tp.lambda_epsilon,  0, 0, "%.2f");
            changed |= ImGui::InputFloat("tensile_k",       &tp.tensile_k,       0, 0, "%.6f");
            changed |= ImGui::InputFloat("tensile_n",       &tp.tensile_n,       0, 0, "%.3f");
            changed |= ImGui::InputFloat("tensile_q",       &tp.tensile_q,       0, 0, "%.3f");
            changed |= ImGui::InputFloat("gravity",         &tp.gravity,         0, 0, "%.3f");
            changed |= ImGui::InputFloat("velocity damping",&tp.velocity_damping,0, 0, "%.4f");
            changed |= ImGui::InputFloat("boundary bounce", &tp.boundary_bounce, 0, 0, "%.3f");
            changed |= ImGui::InputFloat("viscosity_c (XSPH)", &tp.viscosity_c, 0, 0, "%.6f");
            changed |= ImGui::InputFloat("vorticity_eps",   &tp.vorticity_eps,   0, 0, "%.6f");
            changed |= ImGui::InputFloat("max speed",       &tp.max_speed,       0, 0, "%.3f");
            changed |= ImGui::InputFloat("max pos correction", &tp.max_position_correction,
                                         0, 0, "%.5f");
            ImGui::PopItemWidth();
            if (changed) set_tunable_params(tp);

            ImGui::Separator();
            ImGui::Text("camera");
            ImGui::SliderFloat("yaw",      &cam.yaw,      -3.14f, 3.14f, "%.2f");
            ImGui::SliderFloat("pitch",    &cam.pitch,    -1.50f, 1.50f, "%.2f");
            ImGui::SliderFloat("distance", &cam.dist,      1.0f,  80.0f, "%.1f");
            ImGui::SliderFloat("fov",      &cam.fovy_deg, 20.0f, 120.0f, "%.0f°");
            if (ImGui::Button("Reset camera")) { cam = Camera{}; }

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        double render_end = glfwGetTime();
        last_render_ms = static_cast<float>((render_end - render_start) * 1000.0);

        glfwSwapBuffers(win);
    }

    shutdown_simulation();
    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo));
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
