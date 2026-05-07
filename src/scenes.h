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
constexpr int   kSceneParticleCount = 2 << 17;
constexpr float kWorldHalfX        = 6.0f;
constexpr float kWorldHalfY        = 6.0f;
constexpr float kWorldHalfZ        = 3.0f;
constexpr float kParticleRadius    = 0.05f;

const char* scene_name(SceneId id);
void        seed_scene(SceneId id, std::vector<float3>& out);
