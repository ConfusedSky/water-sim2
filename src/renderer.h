#pragma once

#include <glm/glm.hpp>

struct RenderSettings {
    enum Mode { Spheres = 0, ScreenSpace = 1 };
    enum Debug { ComposeFinal = 0, DebugDepth = 1, DebugThickness = 2, DebugNormals = 3 };

    int   mode                 = Spheres;
    int   debug_view           = ComposeFinal;

    int   smooth_iters         = 30;
    int   smooth_radius        = 3;       // bilateral kernel radius (pixels)
    float sigma_spatial        = 2.5f;    // pixels
    float sigma_depth          = 0.50f;   // eye-space units

    float thickness_scale      = 0.04f;   // per-particle contribution
    float thickness_radius_mul = 1.6f;    // sprite radius multiplier vs particle radius

    float refraction_strength  = 0.04f;
    float fresnel_f0           = 0.02f;
    float specular_power       = 96.0f;
    float specular_intensity   = 0.65f;

    float water_color[3]       = {0.16f, 0.46f, 0.72f};
    float absorption[3]        = {0.55f, 0.18f, 0.10f}; // per unit thickness

    bool  show_skybox          = false;
    bool  show_floor           = false;

    float floor_y              = -6.0f;   // world-space height of floor plane
    float floor_cell           = 0.50f;   // world units
    float floor_jitter         = 0.10f;   // 0 = pure checker, 1 = max per-cell variation
    float floor_color_a[3]     = {0.84f, 0.84f, 0.86f};
    float floor_color_b[3]     = {0.32f, 0.32f, 0.34f};

    float sun_dir[3]           = {0.45f, 0.75f, 0.55f};
};

struct RendererInit {
    int   width;
    int   height;
    float particle_radius;
};

class Renderer {
public:
    Renderer();
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void init(const RendererInit& info);
    void shutdown();

    // Renders one frame:
    //   - particle_vao must have a single float4 attribute (xyz=pos, w=density01)
    //   - particle_count is the live particle count
    //   - view, proj come from the camera
    //   - cam_pos is the world-space camera position
    void render(unsigned int particle_vao,
                int           particle_count,
                const glm::mat4& view,
                const glm::mat4& proj,
                const glm::vec3& cam_pos,
                const RenderSettings& s,
                const float bg_color[3]);

    struct Impl;
private:
    Impl* impl_ = nullptr;
};
