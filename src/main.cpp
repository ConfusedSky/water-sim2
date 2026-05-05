#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>
#include <cstdlib>

#include "kernel.cuh"

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
out float v_density;
void main() {
    gl_Position = vec4(in_particle.xy, 0.0, 1.0);
    v_density = in_particle.z;
    float density01 = clamp((in_particle.z - 0.85) / 0.35, 0.0, 1.0);
    gl_PointSize = mix(4.0, 8.0, density01);
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

struct MouseController {
    double cursor_x = 0.0;
    double cursor_y = 0.0;
    bool has_cursor = false;
    bool left_down = false;
    bool right_down = false;
    float radius = 0.14f;
    float strength = 12.0f;
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
    const glm::mat4 projection(1.0f);
    const glm::mat4 inv_projection = glm::inverse(projection);
    glm::vec4 world = inv_projection * glm::vec4(ndc_x, ndc_y, 0.0f, 1.0f);
    return make_float2(world.x / world.w, world.y / world.w);
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

    GLFWwindow* win = glfwCreateWindow(1024, 1024, "water-sim2 — Phase 2", nullptr, nullptr);
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

        glViewport(0, 0, w, h);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, particle_count);
        glBindVertexArray(0);

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
            ImGui::Text("Avg density: %.2f", stats.avg_density);
            ImGui::Text("Max density: %.2f", stats.max_density);
            ImGui::Text("Avg speed: %.3f", stats.avg_speed);
            ImGui::Separator();
            ImGui::Text("Left drag: pull fluid");
            ImGui::Text("Right drag: push fluid");
            ImGui::SliderFloat("Mouse radius", &mouse.radius, 0.05f, 0.30f);
            ImGui::SliderFloat("Mouse strength", &mouse.strength, 2.0f, 32.0f);

            int scene_idx = static_cast<int>(get_active_scene());
            const char* scene_items[kSceneCount] = {
                scene_name(SceneId::ColumnLeft),
                scene_name(SceneId::WideBlock),
                scene_name(SceneId::TwoColumns),
            };
            ImGui::PushItemWidth(160.0f);
            if (ImGui::Combo("scene", &scene_idx, scene_items, kSceneCount)) {
                set_active_scene(static_cast<SceneId>(scene_idx));
                accumulator = kFixedDt;
            }
            ImGui::PopItemWidth();
            if (ImGui::Button("Reset")) {
                reset_simulation();
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
            ImGui::PopItemWidth();
            if (changed) {
                set_tunable_params(tp);
            }

            ImGui::Separator();
            bool use_grid = get_use_spatial_hash();
            if (ImGui::Checkbox("spatial-hash neighbors", &use_grid)) {
                set_use_spatial_hash(use_grid);
            }
            ImGui::TextDisabled("(off = naive O(n^2))");
            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
    }

    CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo));
    shutdown_simulation();
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
