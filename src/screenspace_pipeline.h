#pragma once

#include "pipeline.h"
#include "shader_program.h"

class ScreenspacePipeline : public RenderPipeline {
public:
    explicit ScreenspacePipeline();
    ~ScreenspacePipeline() override;

    ScreenspacePipeline(const ScreenspacePipeline&)            = delete;
    ScreenspacePipeline& operator=(const ScreenspacePipeline&) = delete;

    const char* name() const override { return "screenspace"; }

    bool init(GLuint shared_vbo, int particle_capacity) override;
    void shutdown() override;
    void draw(const Camera& cam, int particle_count, int w, int h) override;
    void reload_shaders() override;
    void draw_imgui_options() override;

private:
    GLuint        vao_    = 0;
    ShaderProgram shader_;
    float         base_color_lo_[3] = {0.06f, 0.28f, 0.65f};
    float         base_color_hi_[3] = {0.50f, 0.82f, 1.00f};
};
