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

void seed_cube_full(std::vector<float3>& out, int& idx) {
    const float side = kWorldHalfX * 2.0f / 3.0f;
    fill_box(out, idx,
             -side * 0.5f, -kWorldHalfY * 0.5f, -side * 0.5f,
             side, side, side);
}

void seed_cube_full_offside(std::vector<float3>& out, int& idx) {
    const float half_w = kWorldHalfX * 2.0f / 3.0f;
    const float half_h = kWorldHalfY * 0.75;
    const float half_d = kWorldHalfZ * 2.0f * 0.9f;
    fill_box(out, idx,
             -kWorldHalfX * 0.9, -kWorldHalfY * 0.9, -half_d * 0.5f,
             half_w, half_h, half_d);
}

void seed_column_left(std::vector<float3>& out, int& idx) {
    const float u = kWorldHalfX / 3.0f;
    const float d = kWorldHalfZ / 3.0f;
    fill_box(out, idx,
             -kWorldHalfX * 0.5f, -kWorldHalfY * 0.5f, -d,
             u, u * 4.0f, d * 2.0f);
}

void seed_wide_block(std::vector<float3>& out, int& idx) {
    const float half_w = kWorldHalfX * 2.0f / 3.0f;
    const float half_d = kWorldHalfZ * 2.0f / 3.0f;
    const float height  = kWorldHalfY / 6.0f;
    fill_box(out, idx,
             -half_w, -kWorldHalfY * 0.5f, -half_d,
             half_w * 2.0f, height, half_d * 2.0f);
}

void seed_large_block(std::vector<float3>& out, int& idx) {
    const float half_w = kWorldHalfX * 2.0f / 3.0f;
    const float half_h = kWorldHalfY * 2.0f / 3.0f;
    const float half_d = kWorldHalfZ * 2.0f / 3.0f;
    fill_box(out, idx,
             -half_w, -half_h, -half_d,
             half_w * 2.0f, half_h, half_d * 2.0f);
}

} // namespace

const char* scene_name(SceneId id) {
    switch (id) {
        case SceneId::CubeFull:   return "cube (full)";
        case SceneId::ColumnLeft: return "column (left)";
        case SceneId::WideBlock:  return "wide block";
        case SceneId::LargeBlock:  return "large block";
        case SceneId::CubeFullOffside:  return "cube (full, offside)";
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
        case SceneId::CubeFullOffside:  seed_cube_full_offside(out, idx);  break;
        default:
            std::fprintf(stderr, "seed_scene: unknown scene id %d\n", static_cast<int>(id));
            std::exit(1);
    }
    out.resize(idx);
}
