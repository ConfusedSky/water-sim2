#pragma once

#include <GL/glew.h>

#include <string>
#include <unordered_map>
#include <sys/types.h>

class ShaderProgram {
public:
    ShaderProgram() = default;
    ~ShaderProgram();

    ShaderProgram(const ShaderProgram&)            = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    // Initial load — returns false and leaves program=0 if compilation fails.
    bool load(std::string vert_path, std::string frag_path);

    // Re-stat the source files; if either changed, recompile.
    // On compile/link failure: prints error, keeps current program live.
    // Returns true iff a successful reload occurred.
    bool reload_if_changed();

    // Force reload regardless of mtime (used by F5).
    bool reload();

    void use() const;
    GLint uniform(const char* name);

    GLuint program() const { return program_; }
    bool   valid()   const { return program_ != 0; }

private:
    bool compile_and_link_();
    static time_t mtime_(const std::string& path);

    std::string vert_path_;
    std::string frag_path_;
    time_t      vert_mtime_ = 0;
    time_t      frag_mtime_ = 0;
    GLuint      program_    = 0;
    std::unordered_map<std::string, GLint> uniform_cache_;
};
