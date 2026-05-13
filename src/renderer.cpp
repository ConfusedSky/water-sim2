#include "renderer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <limits.h>

namespace {

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

std::string exe_dir() {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    std::string p(buf);
    auto slash = p.find_last_of('/');
    return (slash == std::string::npos) ? "." : p.substr(0, slash);
}

std::string shaders_dir() {
    if (const char* env = std::getenv("SHADERS_DIR")) {
        if (env[0] != '\0') return env;
    }
    return exe_dir() + "/shaders";
}

}  // namespace

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

glm::vec3 Camera::position() const {
    return target + glm::vec3(
        cosf(pitch) * sinf(yaw),
        sinf(pitch),
        cosf(pitch) * cosf(yaw)) * dist;
}

glm::mat4 Camera::view() const {
    return glm::lookAt(position(), target, glm::vec3(0, 1, 0));
}

glm::mat4 Camera::proj(float aspect) const {
    return glm::perspective(glm::radians(fovy_deg), aspect, near_z, far_z);
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::init(int particle_capacity) {
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 particle_capacity * sizeof(float) * 4,
                 nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, nullptr);
    glBindVertexArray(0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_, vbo_,
                                            cudaGraphicsRegisterFlagsNone));

    std::string dir = shaders_dir();
    std::string vs_path = dir + "/particle.vert";
    std::string fs_path = dir + "/particle.frag";
    std::printf("shaders: %s\n", dir.c_str());
    if (!shader_.load(vs_path, fs_path)) {
        std::fprintf(stderr, "shader: initial load failed\n");
        return false;
    }
    return true;
}

void Renderer::shutdown() {
    if (cuda_vbo_) {
        cudaGraphicsUnregisterResource(cuda_vbo_);
        cuda_vbo_ = nullptr;
    }
    if (vbo_) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
}

float4* Renderer::map_vbo() {
    float4* dptr = nullptr;
    size_t bytes = 0;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo_, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&dptr), &bytes, cuda_vbo_));
    return dptr;
}

void Renderer::unmap_vbo() {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_, 0));
}

void Renderer::request_reload() {
    if (shader_.reload()) {
        std::printf("shaders: reloaded\n");
    } else {
        std::fprintf(stderr, "shaders: reload failed (keeping previous program)\n");
    }
}

void Renderer::draw(const Camera& cam, int particle_count, int w, int h,
                    const float bg[3]) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, w, h);
    glClearColor(bg[0], bg[1], bg[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!shader_.valid()) return;

    float aspect = (h > 0) ? static_cast<float>(w) / h : 1.0f;
    glm::mat4 proj_mat = cam.proj(aspect);
    glm::mat4 view_mat = cam.view();
    glm::mat4 mvp      = proj_mat * view_mat;

    glDisable(GL_BLEND);
    shader_.use();
    glUniformMatrix4fv(shader_.uniform("u_mv"),  1, GL_FALSE, glm::value_ptr(view_mat));
    glUniformMatrix4fv(shader_.uniform("u_mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(shader_.uniform("u_radius_world"),  kParticleRadius);
    glUniform1f(shader_.uniform("u_proj11"),        proj_mat[1][1]);
    glUniform1f(shader_.uniform("u_viewport_h"),    static_cast<float>(h));
    glUniform1f(shader_.uniform("u_sphere_radius"), kParticleRadius);
    glUniform1f(shader_.uniform("u_proj22"),        proj_mat[2][2]);
    glUniform1f(shader_.uniform("u_proj32"),        proj_mat[3][2]);

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, particle_count);
    glBindVertexArray(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}
