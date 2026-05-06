#pragma once

#include <vector>
#include <vector_types.h>

enum class SceneId {
    CubeFull  = 0,
    ColumnLeft = 1,
    WideBlock  = 2,
    LargeBlock  = 3,
};

constexpr int   kSceneCount        = 4;
constexpr int   kSceneParticleCount = 2 << 15;
constexpr float kWorldHalfExtent   = 3.0f;
constexpr float kParticleRadius    = 0.05f;

const char* scene_name(SceneId id);
void        seed_scene(SceneId id, std::vector<float3>& out);
