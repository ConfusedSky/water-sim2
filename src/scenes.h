#pragma once

#include <vector>

#include <vector_types.h>

enum class SceneId {
    ColumnLeft = 0,
    WideBlock = 1,
    TwoColumns = 2,
};

constexpr int kSceneCount = 3;
constexpr int kMaxParticleCount = 65536;
constexpr float kWorldHalfExtent = 3.0f;
constexpr float kDefaultSpacing = 0.019f;

const char* scene_name(SceneId id);

void seed_scene(SceneId id, std::vector<float2>& out);
