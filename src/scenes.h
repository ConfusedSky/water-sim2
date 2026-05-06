#pragma once

#include <vector>
#include <vector_types.h>

enum class SceneId {
    CubeFull  = 0,
    ColumnLeft = 1,
    WideBlock  = 2,
};

constexpr int   kSceneCount        = 3;
constexpr int   kSceneParticleCount = 32768;
constexpr float kWorldHalfExtent   = 4.0f;

const char* scene_name(SceneId id);
void        seed_scene(SceneId id, std::vector<float3>& out);
