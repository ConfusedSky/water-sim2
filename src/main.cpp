#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>
#include <cstdlib>

#include "kernel.cuh"

constexpr float kFixedDt = 1.0f / 240.0f;
constexpr int kMaxSubsteps = 16;

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

int main() {
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(1024, 1024, "water-sim2 — Phase 1", nullptr, nullptr);
    if (!win) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

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

        double now = glfwGetTime();
        double frame_dt = now - last_time;
        last_time = now;
        if (frame_dt > 0.05) {
            frame_dt = 0.05;
        }
        accumulator += frame_dt;

        float4* dptr = nullptr;
        size_t bytes = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dptr), &bytes, cuda_vbo));
        int substeps = 0;
        while (accumulator >= kFixedDt && substeps < kMaxSubsteps) {
            step_simulation(kFixedDt, dptr);
            accumulator -= kFixedDt;
            ++substeps;
        }
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo, 0));
        stats = get_simulation_stats();

        int w = 0, h = 0;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, particle_count);
        glBindVertexArray(0);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
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
            if (ImGui::Button("Reset")) {
                reset_simulation();
                accumulator = kFixedDt;
            }
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
