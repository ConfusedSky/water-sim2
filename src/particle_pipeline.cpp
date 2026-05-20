#include "particle_pipeline.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#include <cstdio>
#include <string>

ParticlePipeline::ParticlePipeline(std::string name) : name_(std::move(name)) {}

ParticlePipeline::~ParticlePipeline() {
    shutdown();
}

bool ParticlePipeline::init(GLuint shared_vbo, int /*particle_capacity*/) {
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, shared_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float) * 4, nullptr);
    glBindVertexArray(0);

    std::string dir = shaders_dir();
    std::string vs_path = dir + "/particle.vert";
    std::string fs_path = dir + "/particle.frag";
    if (!shader_.load(vs_path, fs_path)) {
        std::fprintf(stderr, "shader: initial load failed for pipeline '%s'\n",
                     name_.c_str());
        return false;
    }
    return true;
}

void ParticlePipeline::shutdown() {
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
}

void ParticlePipeline::draw(const Camera& cam, int particle_count, int w, int h) {
    if (!shader_.valid()) return;

    float aspect = (h > 0) ? static_cast<float>(w) / h : 1.0f;
    glm::mat4 proj_mat = cam.proj(aspect);
    glm::mat4 view_mat = cam.view();
    glm::mat4 mvp      = proj_mat * view_mat;

    glDisable(GL_BLEND);
    shader_.use();
    glUniformMatrix4fv(shader_.uniform("u_mv"),  1, GL_FALSE, glm::value_ptr(view_mat));
    glUniformMatrix4fv(shader_.uniform("u_mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1f(shader_.uniform("u_particle_radius"), Renderer::kParticleRadius);
    glUniform1f(shader_.uniform("u_proj11"),          proj_mat[1][1]);
    glUniform1f(shader_.uniform("u_viewport_h"),      static_cast<float>(h));
    glUniform1f(shader_.uniform("u_proj22"),          proj_mat[2][2]);
    glUniform1f(shader_.uniform("u_proj32"),          proj_mat[3][2]);
    glUniform3fv(shader_.uniform("u_base_color_lo"), 1, base_color_lo_);
    glUniform3fv(shader_.uniform("u_base_color_hi"), 1, base_color_hi_);

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, particle_count);
    glBindVertexArray(0);
}

void ParticlePipeline::reload_shaders() {
    if (shader_.reload()) {
        std::printf("shaders: reloaded for pipeline '%s'\n", name_.c_str());
    } else {
        std::fprintf(stderr,
                     "shaders: reload failed for pipeline '%s' (keeping previous program)\n",
                     name_.c_str());
    }
}

void ParticlePipeline::draw_imgui_options() {
    ImGui::ColorEdit3("base color (low density)",  base_color_lo_);
    ImGui::ColorEdit3("base color (high density)", base_color_hi_);
}
