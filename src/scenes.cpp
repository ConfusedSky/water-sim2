#include "scenes.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr float kSpacing = kParticleRadius * 2.0f;
constexpr float kJitter  = kSpacing * 0.15f;

// Integer floor cube-root (C++17 constexpr)
constexpr int icbrt(int n) {
    int x = 1;
    while ((x + 1) * (x + 1) * (x + 1) <= n) ++x;
    return x;
}

// Each scene keeps a fixed aspect ratio; +1 ensures the grid holds >= N particles.
// fill_box will stop once out.size() particles are written.

// CubeFull: 1 : 1 : 1
constexpr int kCubeSide = icbrt(kSceneParticleCount) + 1;

// ColumnLeft: width : height : depth = 1 : 4 : 2
constexpr int kColBase  = icbrt(kSceneParticleCount / 8) + 1;
constexpr int kColW     = kColBase;
constexpr int kColH     = kColBase * 4;
constexpr int kColD     = kColBase * 2;

// WideBlock: width : height : depth = 8 : 1 : 8
constexpr int kWideBase = icbrt(kSceneParticleCount / 64) + 1;
constexpr int kWideW    = kWideBase * 8;
constexpr int kWideH    = kWideBase;
constexpr int kWideD    = kWideBase * 8;

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

// Centred cube
void seed_cube_full(std::vector<float3>& out) {
    int idx = 0;
    float half = kCubeSide * kSpacing * 0.5f;
    fill_box(out, idx,
             -half,
             -kWorldHalfExtent * 0.5f,
             -half,
             kCubeSide, kCubeSide, kCubeSide);
}

// Tall column on the left
void seed_column_left(std::vector<float3>& out) {
    int idx = 0;
    fill_box(out, idx,
             -kWorldHalfExtent * 0.5f,
             -kWorldHalfExtent * 0.5f,
             -kColD * kSpacing * 0.5f,
             kColW, kColH, kColD);
}

// Wide shallow layer at the bottom
void seed_wide_block(std::vector<float3>& out) {
    int idx = 0;
    fill_box(out, idx,
             -kWideW * kSpacing * 0.5f,
             -kWorldHalfExtent * 0.5f,
             -kWideD * kSpacing * 0.5f,
             kWideW, kWideH, kWideD);
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
