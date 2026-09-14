#pragma once

#include <GL/glew.h>

#include "renderer.h"

class RenderPipeline {
public:
    virtual ~RenderPipeline() = default;

    virtual const char* name() const = 0;

    // Build VAO/shaders against the renderer's shared particle VBO.
    virtual bool init(GLuint shared_vbo, int particle_capacity) = 0;
    virtual void shutdown() = 0;

    // Issue draw calls. The framebuffer is already cleared by Renderer::draw().
    virtual void draw(const Camera& cam, int particle_count, int w, int h) = 0;

    virtual void reload_shaders() = 0;

    // Pipeline-specific controls inside the renderer ImGui window. Default: empty.
    virtual void draw_imgui_options() {}
};
