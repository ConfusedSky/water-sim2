#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>

#include "imgui_widgets.h"
#include "kernel.cuh"
#include "particle_pipeline.h"
#include "pipeline.h"
#include "renderer.h"
#include "screenspace_pipeline.h"

#include <memory>

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

constexpr float kFixedDt     = 1.0f / 120.0f;
constexpr int   kMaxSubsteps = 4;
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

struct AppContext {
    InputState* input    = nullptr;
    Renderer*   renderer = nullptr;
};

static void cursor_position_callback(GLFWwindow* win, double x, double y) {
    auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(win));
    if (!ctx || !ctx->input) return;
    ctx->input->cursor_x   = x;
    ctx->input->cursor_y   = y;
    ctx->input->has_cursor = true;
}

static void mouse_button_callback(GLFWwindow* win, int button, int action, int) {
    auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(win));
    if (!ctx || !ctx->input) return;
    bool down = (action != GLFW_RELEASE);
    if (button == GLFW_MOUSE_BUTTON_LEFT)  ctx->input->left_down  = down;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) ctx->input->right_down = down;
}

static void scroll_callback(GLFWwindow* win, double, double yoff) {
    auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(win));
    if (ctx && ctx->input) ctx->input->scroll_dy += static_cast<float>(yoff);
}

static void key_callback(GLFWwindow* win, int key, int, int action, int) {
    auto* ctx = static_cast<AppContext*>(glfwGetWindowUserPointer(win));
    if (!ctx) return;
    if (key == GLFW_KEY_F5 && action == GLFW_PRESS && ctx->renderer) {
        ctx->renderer->request_reload();
    }
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
    Renderer   renderer{};
    AppContext ctx{&inp, &renderer};
    glfwSetWindowUserPointer(win, &ctx);
    glfwSetCursorPosCallback(win,  cursor_position_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetScrollCallback(win,      scroll_callback);
    glfwSetKeyCallback(win,         key_callback);

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

    if (!renderer.init(kSceneParticleCount)) {
        std::fprintf(stderr, "renderer.init failed\n");
        return 1;
    }
    if (!renderer.add_pipeline(std::make_unique<ParticlePipeline>()) ||
        !renderer.add_pipeline(std::make_unique<ScreenspacePipeline>())) {
        std::fprintf(stderr, "renderer: pipeline registration failed\n");
        return 1;
    }

    // --- State ---------------------------------------------------------------
    Camera cam{};
    SceneId active_scene = SceneId::CubeFullOffside;
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
        float4* dptr = renderer.map_vbo();
        int substeps = 0;
        while (accumulator >= kFixedDt && substeps < kMaxSubsteps) {
            step_simulation(kFixedDt, sim_mouse, dptr);
            accumulator -= kFixedDt;
            ++substeps;
        }
        renderer.unmap_vbo();
        stats = get_simulation_stats();
        double sim_end = glfwGetTime();
        float sim_ms = static_cast<float>((sim_end - sim_start) * 1000.0);
        frame_window.push(sim_end, sim_ms, last_render_ms);

        // --- Render ----------------------------------------------------------
        double render_start = glfwGetTime();
        renderer.draw(cam, get_particle_count(), w, h, bg_color);

        // --- ImGui -----------------------------------------------------------
        {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
            ImGui::Begin("stats", nullptr,
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoNav);
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("CUDA: %s", prop.name);
            ImGui::Text("Particles: %d", get_particle_count());
            ImGui::Text("Solver iterations: %d", get_solver_iterations());
            float as = 0, ms = 0, ar = 0, mr = 0;
            frame_window.summarize(as, ms, ar, mr);
            ImGui::Text("Sim    avg/max: %.2f / %.2f ms", as, ms);
            ImGui::Text("Render avg/max: %.2f / %.2f ms", ar, mr);
            ImGui::Text("Avg density: %.2f", stats.avg_density);
            ImGui::Text("Max density: %.2f", stats.max_density);
            ImGui::Text("Avg speed: %.3f",   stats.avg_speed);
            ImGui::Separator();
            ImGui::Text("Left drag: orbit  |  Scroll: zoom");
            ImGui::Text("Right drag: push fluid");
            ImGui::Text("F5: reload shaders");
            ImGui::SliderFloat("Mouse radius",   &inp.mouse_radius,   0.1f, 2.0f);
            ImGui::SliderFloat("Mouse strength", &inp.mouse_strength, 5.0f, 500.0f);

            const char* scene_names[] = { "cube (full)", "column (left)", "wide block", "large block", "cube (full, offside)" };
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
            changed |= ui::ProportionalDragFloat("rest density",       &tp.rest_density,       "%.2f");
            changed |= ui::ProportionalDragFloat("kernel radius",      &tp.kernel_radius,      "%.4f");
            changed |= ImGui::InputInt("solver iterations", &tp.solver_iterations, 0, 0);
            changed |= ui::ProportionalDragFloat("lambda_epsilon",     &tp.lambda_epsilon,     "%.2f");
            changed |= ui::ProportionalDragFloat("tensile_k",          &tp.tensile_k,          "%.6f");
            changed |= ui::ProportionalDragFloat("tensile_n",          &tp.tensile_n,          "%.3f");
            changed |= ui::ProportionalDragFloat("tensile_q",          &tp.tensile_q,          "%.3f");
            changed |= ui::ProportionalDragFloat("gravity",            &tp.gravity,            "%.3f");
            changed |= ui::ProportionalDragFloat("velocity damping",   &tp.velocity_damping,   "%.4f");
            changed |= ui::ProportionalDragFloat("boundary bounce",    &tp.boundary_bounce,    "%.3f");
            changed |= ui::ProportionalDragFloat("viscosity_c (XSPH)", &tp.viscosity_c,        "%.6f");
            changed |= ui::ProportionalDragFloat("vorticity_eps",      &tp.vorticity_eps,      "%.6f");
            changed |= ui::ProportionalDragFloat("max speed",          &tp.max_speed,          "%.3f");
            changed |= ui::ProportionalDragFloat("max pos correction", &tp.max_position_correction, "%.5f");
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

        {
            ImGui::SetNextWindowPos(ImVec2(static_cast<float>(w) - 10.0f, 360.0f),
                                    ImGuiCond_FirstUseEver, ImVec2(1.0f, 0.0f));
            ImGui::SetNextWindowSize(ImVec2(280.0f, 0.0f), ImGuiCond_FirstUseEver);
            ImGui::Begin("renderer");

            int n = renderer.pipeline_count();
            if (n > 0) {
                int active = renderer.active_pipeline_index();
                if (ImGui::BeginCombo("pipeline", renderer.pipeline_name(active))) {
                    for (int i = 0; i < n; ++i) {
                        bool selected = (i == active);
                        if (ImGui::Selectable(renderer.pipeline_name(i), selected)) {
                            renderer.set_active_pipeline(i);
                        }
                        if (selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::Separator();
                if (auto* p = renderer.active_pipeline()) {
                    p->draw_imgui_options();
                }
            } else {
                ImGui::TextUnformatted("no pipelines registered");
            }

            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        double render_end = glfwGetTime();
        last_render_ms = static_cast<float>((render_end - render_start) * 1000.0);

        glfwSwapBuffers(win);
    }

    shutdown_simulation();
    renderer.shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
