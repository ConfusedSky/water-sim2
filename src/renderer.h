#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector_types.h>

#include <memory>
#include <vector>

struct cudaGraphicsResource;
class  RenderPipeline;

struct Camera {
    float     yaw      = 0.5f;
    float     pitch    = 0.3f;
    float     dist     = 10.0f;
    glm::vec3 target   = {0.0f, -5.0f, 0.0f};
    float     fovy_deg = 45.0f;
    float     near_z   = 0.1f;
    float     far_z    = 200.0f;

    glm::vec3 position() const;
    glm::mat4 view()     const;
    glm::mat4 proj(float aspect) const;
};

class Renderer {
public:
    static constexpr float kParticleRadius = 0.05f;

    Renderer();
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Allocates the shared CUDA-interop VBO and registers it. Pipelines are
    // added afterwards via add_pipeline().
    bool init(int particle_capacity);
    void shutdown();

    // Registers a pipeline and immediately initializes it against the shared VBO.
    // Returns false (and discards the pipeline) on init failure.
    bool add_pipeline(std::unique_ptr<RenderPipeline> pipeline);

    int             pipeline_count() const { return static_cast<int>(pipelines_.size()); }
    const char*     pipeline_name(int i) const;
    int             active_pipeline_index() const { return active_index_; }
    void            set_active_pipeline(int i);
    RenderPipeline* active_pipeline();

    // CUDA interop — map/unmap the VBO around step_simulation().
    float4* map_vbo();
    void    unmap_vbo();

    // Reload shaders for the active pipeline only (F5).
    void request_reload();

    // Clears the framebuffer, then dispatches to the active pipeline.
    void draw(const Camera& cam, int particle_count, int w, int h, const float bg[3]);

private:
    GLuint                vbo_                = 0;
    cudaGraphicsResource* cuda_vbo_           = nullptr;
    int                   particle_capacity_  = 0;

    std::vector<std::unique_ptr<RenderPipeline>> pipelines_;
    int                                          active_index_ = 0;
};
