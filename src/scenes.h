#pragma once

#include <vector>

#include <vector_types.h>

enum class SceneId {
    ColumnLeft = 0,
    WideBlock = 1,
    TwoColumns = 2,
};

constexpr int kSceneCount = 3;
constexpr int kSceneParticleCount = 2048;

const char* scene_name(SceneId id);

void seed_scene(SceneId id, std::vector<float2>& out);
