#include "scenes.h"

#include <cstdio>
#include <cstdlib>

namespace {

constexpr float kSpacing = 0.019f;
constexpr float kRowJitter = kSpacing * 0.15f;

void fill_grid(std::vector<float2>& out, int& write_idx, float start_x,
               float start_y, int cols, int rows) {
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float offset_x = (y & 1) ? kRowJitter : 0.0f;
            out[write_idx++] = float2{start_x + x * kSpacing + offset_x,
                                      start_y + y * kSpacing};
        }
    }
}

void seed_column_left(std::vector<float2>& out) {
    int idx = 0;
    fill_grid(out, idx, -2.70f, -2.70f, 64, 256);
}

void seed_wide_block(std::vector<float2>& out) {
    int idx = 0;
    fill_grid(out, idx, -2.432f, -2.70f, 256, 64);
}

void seed_two_columns(std::vector<float2>& out) {
    int idx = 0;
    fill_grid(out, idx, -2.70f, -2.70f, 32, 256);
    fill_grid(out, idx, 2.092f, -2.70f, 32, 256);
}

}  // namespace

const char* scene_name(SceneId id) {
    switch (id) {
        case SceneId::ColumnLeft:
            return "column (left)";
        case SceneId::WideBlock:
            return "wide block (bottom)";
        case SceneId::TwoColumns:
            return "two columns";
    }
    return "?";
}

void seed_scene(SceneId id, std::vector<float2>& out) {
    out.assign(kMaxParticleCount, float2{0.0f, 0.0f});
    switch (id) {
        case SceneId::ColumnLeft:
            seed_column_left(out);
            return;
        case SceneId::WideBlock:
            seed_wide_block(out);
            return;
        case SceneId::TwoColumns:
            seed_two_columns(out);
            return;
    }
    std::fprintf(stderr, "seed_scene: unknown scene id %d\n", static_cast<int>(id));
    std::exit(1);
}
