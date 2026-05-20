#include "screenspace_pipeline.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#include <cstdio>
#include <string>

ScreenspacePipeline::ScreenspacePipeline() {}

ScreenspacePipeline::~ScreenspacePipeline() {
    shutdown();
}

bool ScreenspacePipeline::init(GLuint shared_vbo, int /*particle_capacity*/) {
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
                     name());
        return false;
    }
    return true;
}

void ScreenspacePipeline::shutdown() {
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
}

void ScreenspacePipeline::draw(const Camera& cam, int particle_count, int w, int h) {
    if (!shader_.valid()) return;

    float aspect = (h > 0) ? static_cast<float>(w) / h : 1.0f;
    glm::mat4 proj_mat = cam.proj(aspect);
    glm::mat4 view_mat = cam.view();

    glDisable(GL_BLEND);
    shader_.use();
    glUniformMatrix4fv(shader_.uniform("u_mv"),   1, GL_FALSE, glm::value_ptr(view_mat));
    glUniformMatrix4fv(shader_.uniform("u_proj"), 1, GL_FALSE, glm::value_ptr(proj_mat));
    glUniform1f(shader_.uniform("u_particle_radius"), Renderer::kParticleRadius);
    glUniform1f(shader_.uniform("u_viewport_h"),      static_cast<float>(h));
    glUniform3fv(shader_.uniform("u_base_color_lo"), 1, base_color_lo_);
    glUniform3fv(shader_.uniform("u_base_color_hi"), 1, base_color_hi_);

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, particle_count);
    glBindVertexArray(0);
}

void ScreenspacePipeline::reload_shaders() {
    if (shader_.reload()) {
        std::printf("shaders: reloaded for pipeline '%s'\n", name());
    } else {
        std::fprintf(stderr,
                     "shaders: reload failed for pipeline '%s' (keeping previous program)\n",
                     name());
    }
}

void ScreenspacePipeline::draw_imgui_options() {
    ImGui::ColorEdit3("base color (low density)",  base_color_lo_);
    ImGui::ColorEdit3("base color (high density)", base_color_hi_);
}
