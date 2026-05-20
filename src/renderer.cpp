#include "renderer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "pipeline.h"

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

Renderer::Renderer() = default;

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::init(int particle_capacity) {
    particle_capacity_ = particle_capacity;

    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 particle_capacity * sizeof(float) * 4,
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_, vbo_,
                                            cudaGraphicsRegisterFlagsNone));
    return true;
}

void Renderer::shutdown() {
    pipelines_.clear();
    if (cuda_vbo_) {
        cudaGraphicsUnregisterResource(cuda_vbo_);
        cuda_vbo_ = nullptr;
    }
    if (vbo_) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
}

bool Renderer::add_pipeline(std::unique_ptr<RenderPipeline> pipeline) {
    if (!pipeline) return false;
    if (!pipeline->init(vbo_, particle_capacity_)) {
        std::fprintf(stderr, "renderer: pipeline '%s' init failed\n",
                     pipeline->name());
        return false;
    }
    pipelines_.push_back(std::move(pipeline));
    return true;
}

const char* Renderer::pipeline_name(int i) const {
    if (i < 0 || i >= static_cast<int>(pipelines_.size())) return "";
    return pipelines_[i]->name();
}

void Renderer::set_active_pipeline(int i) {
    if (i < 0 || i >= static_cast<int>(pipelines_.size())) return;
    active_index_ = i;
}

RenderPipeline* Renderer::active_pipeline() {
    if (pipelines_.empty()) return nullptr;
    if (active_index_ < 0 ||
        active_index_ >= static_cast<int>(pipelines_.size())) return nullptr;
    return pipelines_[active_index_].get();
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
    if (auto* p = active_pipeline()) p->reload_shaders();
}

void Renderer::draw(const Camera& cam, int particle_count, int w, int h,
                    const float bg[3]) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, w, h);
    glClearColor(bg[0], bg[1], bg[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (auto* p = active_pipeline()) p->draw(cam, particle_count, w, h);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}
