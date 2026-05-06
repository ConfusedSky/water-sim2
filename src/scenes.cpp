#include "scenes.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr float kSpacing = kParticleRadius * 2.0f;
constexpr float kJitter  = kSpacing * 0.15f;

void fill_box(std::vector<float3>& out, int& idx,
              float ox, float oy, float oz,
              float width, float height, float depth_extent) {
    int cols  = static_cast<int>(width        / kSpacing);
    int rows  = static_cast<int>(height       / kSpacing);
    int slabs = static_cast<int>(depth_extent / kSpacing);
    for (int z = 0; z < slabs && idx < (int)out.size(); ++z)
        for (int y = 0; y < rows && idx < (int)out.size(); ++y) {
            float jx = (y & 1) ? kJitter : 0.0f;
            for (int x = 0; x < cols && idx < (int)out.size(); ++x)
                out[idx++] = make_float3(ox + x * kSpacing + jx,
                                         oy + y * kSpacing,
                                         oz + z * kSpacing);
        }
}

// Centred cube, 2/3 W per side  →  40×40×40 = 64 000 particles at W=6
void seed_cube_full(std::vector<float3>& out, int& idx) {
    const float side = kWorldHalfExtent * 2.0f / 3.0f;
    fill_box(out, idx,
             -side * 0.5f, -kWorldHalfExtent * 0.5f, -side * 0.5f,
             side, side, side);
}

// Tall column on the left, 1:4:2 ratio  →  20×80×40 = 64 000 particles at W=6
void seed_column_left(std::vector<float3>& out, int& idx) {
    const float u = kWorldHalfExtent / 3.0f;
    fill_box(out, idx,
             -kWorldHalfExtent * 0.5f, -kWorldHalfExtent * 0.5f, -u,
             u, u * 4.0f, u * 2.0f);
}

// Wide shallow slab, 8:1:8 ratio  →  80×10×80 = 64 000 particles at W=6
void seed_wide_block(std::vector<float3>& out, int& idx) {
    const float half_w = kWorldHalfExtent * 2.0f / 3.0f;
    const float height  = kWorldHalfExtent / 6.0f;
    fill_box(out, idx,
             -half_w, -kWorldHalfExtent * 0.5f, -half_w,
             half_w * 2.0f, height, half_w * 2.0f);
}

void seed_large_block(std::vector<float3>& out, int& idx) {
    const float half_w = kWorldHalfExtent * 2.0f / 3.0f;
    fill_box(out, idx,
             -half_w, -half_w, -half_w,
             half_w * 2.0f, half_w * 2, half_w * 2.0f);
}

} // namespace

const char* scene_name(SceneId id) {
    switch (id) {
        case SceneId::CubeFull:   return "cube (full)";
        case SceneId::ColumnLeft: return "column (left)";
        case SceneId::WideBlock:  return "wide block";
        case SceneId::LargeBlock:  return "large block";
    }
    return "?";
}

void seed_scene(SceneId id, std::vector<float3>& out) {
    out.assign(kSceneParticleCount, make_float3(0.0f, 0.0f, 0.0f));
    int idx = 0;
    switch (id) {
        case SceneId::CubeFull:   seed_cube_full(out, idx);   break;
        case SceneId::ColumnLeft: seed_column_left(out, idx); break;
        case SceneId::WideBlock:  seed_wide_block(out, idx);  break;
        case SceneId::LargeBlock:  seed_large_block(out, idx);  break;
        default:
            std::fprintf(stderr, "seed_scene: unknown scene id %d\n", static_cast<int>(id));
            std::exit(1);
    }
    out.resize(idx);
}
