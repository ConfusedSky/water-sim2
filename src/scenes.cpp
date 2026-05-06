#include "scenes.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr float kSpacing = 0.10f;
constexpr float kJitter  = kSpacing * 0.15f;

void fill_box(std::vector<float3>& out, int& idx,
              float ox, float oy, float oz,
              int cols, int rows, int depth) {
    for (int z = 0; z < depth && idx < (int)out.size(); ++z)
        for (int y = 0; y < rows && idx < (int)out.size(); ++y) {
            float jx = (y & 1) ? kJitter : 0.0f;
            for (int x = 0; x < cols && idx < (int)out.size(); ++x)
                out[idx++] = make_float3(ox + x * kSpacing + jx,
                                         oy + y * kSpacing,
                                         oz + z * kSpacing);
        }
}

// 32×32×32 = 32768 — centred cube
void seed_cube_full(std::vector<float3>& out) {
    int idx = 0;
    fill_box(out, idx, -1.6f, -2.8f, -1.6f, 32, 32, 32);
}

// 16×64×32 = 32768 — tall column on the left
void seed_column_left(std::vector<float3>& out) {
    int idx = 0;
    fill_box(out, idx, -2.8f, -2.9f, -1.6f, 16, 64, 32);
}

// 64×8×64 = 32768 — wide shallow layer at the bottom
void seed_wide_block(std::vector<float3>& out) {
    int idx = 0;
    fill_box(out, idx, -3.2f, -2.9f, -3.2f, 64, 8, 64);
}

} // namespace

const char* scene_name(SceneId id) {
    switch (id) {
        case SceneId::CubeFull:   return "cube (full)";
        case SceneId::ColumnLeft: return "column (left)";
        case SceneId::WideBlock:  return "wide block";
    }
    return "?";
}

void seed_scene(SceneId id, std::vector<float3>& out) {
    out.assign(kSceneParticleCount, make_float3(0.0f, 0.0f, 0.0f));
    switch (id) {
        case SceneId::CubeFull:   seed_cube_full(out);   return;
        case SceneId::ColumnLeft: seed_column_left(out); return;
        case SceneId::WideBlock:  seed_wide_block(out);  return;
    }
    std::fprintf(stderr, "seed_scene: unknown scene id %d\n", static_cast<int>(id));
    std::exit(1);
}
