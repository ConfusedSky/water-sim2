#pragma once

#include <string>

#include "pipeline.h"
#include "shader_program.h"

class ParticlePipeline : public RenderPipeline {
public:
    explicit ParticlePipeline(std::string name);
    ~ParticlePipeline() override;

    ParticlePipeline(const ParticlePipeline&)            = delete;
    ParticlePipeline& operator=(const ParticlePipeline&) = delete;

    const char* name() const override { return name_.c_str(); }

    bool init(GLuint shared_vbo, int particle_capacity) override;
    void shutdown() override;
    void draw(const Camera& cam, int particle_count, int w, int h) override;
    void reload_shaders() override;
    void draw_imgui_options() override;

private:
    std::string   name_;
    GLuint        vao_    = 0;
    ShaderProgram shader_;
    float         base_color_lo_[3] = {0.06f, 0.28f, 0.65f};
    float         base_color_hi_[3] = {0.50f, 0.82f, 1.00f};
};
