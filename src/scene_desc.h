#pragma once

#include <string>
#include <vector>
#include <vector_types.h>

enum class ObstacleType { Circle, Box };

struct ObstacleCircle { float cx, cy, r; };
struct ObstacleBox    { float cx, cy, hw, hh; };

struct Obstacle {
    ObstacleType type;
    ObstacleCircle circle{};
    ObstacleBox    box{};
};

struct SpawnRect {
    float x, y, width, height;
};

struct SceneDesc {
    std::string            name;
    float                  spacing = 0.019f;
    std::vector<SpawnRect> spawn_rects;
    std::vector<Obstacle>  obstacles;
};

constexpr int kSdfResolution = 512;

bool load_scene_json(const std::string& path, SceneDesc& out, std::string& error);
std::vector<std::string> list_scene_files(const std::string& dir);
void bake_sdf(const SceneDesc& scene, float world_half, int resolution,
              std::vector<float>& out_pixels);
void seed_from_scene_desc(const SceneDesc& scene, std::vector<float2>& out);
