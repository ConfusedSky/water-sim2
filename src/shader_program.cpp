#include "shader_program.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace {

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "shader: cannot open '%s'\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

GLuint compile(GLenum type, const char* src, const char* tag) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "shader compile error (%s): %s\n", tag, log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint link(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "shader link error: %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

}  // namespace

ShaderProgram::~ShaderProgram() {
    if (program_) glDeleteProgram(program_);
}

time_t ShaderProgram::mtime_(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) return 0;
    return st.st_mtime;
}

bool ShaderProgram::compile_and_link_() {
    std::string vs_src, fs_src;
    if (!read_file(vert_path_, vs_src)) return false;
    if (!read_file(frag_path_, fs_src)) return false;

    GLuint vs = compile(GL_VERTEX_SHADER,   vs_src.c_str(), vert_path_.c_str());
    GLuint fs = compile(GL_FRAGMENT_SHADER, fs_src.c_str(), frag_path_.c_str());
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return false;
    }

    GLuint new_prog = link(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!new_prog) return false;

    if (program_) glDeleteProgram(program_);
    program_ = new_prog;
    uniform_cache_.clear();
    vert_mtime_ = mtime_(vert_path_);
    frag_mtime_ = mtime_(frag_path_);
    return true;
}

bool ShaderProgram::load(std::string vert_path, std::string frag_path) {
    vert_path_ = std::move(vert_path);
    frag_path_ = std::move(frag_path);
    return compile_and_link_();
}

bool ShaderProgram::reload() {
    return compile_and_link_();
}

bool ShaderProgram::reload_if_changed() {
    time_t v = mtime_(vert_path_);
    time_t f = mtime_(frag_path_);
    if (v == vert_mtime_ && f == frag_mtime_) return false;
    return compile_and_link_();
}

void ShaderProgram::use() const {
    glUseProgram(program_);
}

GLint ShaderProgram::uniform(const char* name) {
    auto it = uniform_cache_.find(name);
    if (it != uniform_cache_.end()) return it->second;
    GLint loc = glGetUniformLocation(program_, name);
    uniform_cache_.emplace(name, loc);
    return loc;
}
