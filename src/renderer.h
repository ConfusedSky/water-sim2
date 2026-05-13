#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector_types.h>

#include "shader_program.h"

struct cudaGraphicsResource;

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

    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Allocates VBO/VAO, registers it with CUDA, loads shaders.
    // Returns false if shader compilation fails (caller may still proceed —
    // shaders can be reloaded later via request_reload()).
    bool init(int particle_capacity);
    void shutdown();

    // CUDA interop — map/unmap the VBO around step_simulation().
    float4* map_vbo();
    void    unmap_vbo();

    // Manual reload (e.g. F5 key). Logs result.
    void request_reload();

    // Draws particles. Clears the framebuffer first.
    void draw(const Camera& cam, int particle_count, int w, int h, const float bg[3]);

private:
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    cudaGraphicsResource* cuda_vbo_ = nullptr;
    ShaderProgram shader_;
};
