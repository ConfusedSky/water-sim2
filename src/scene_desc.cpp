#include "scene_desc.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

static float sdf_circle_eval(float px, float py, const ObstacleCircle& c) {
    float dx = px - c.cx, dy = py - c.cy;
    return sqrtf(dx * dx + dy * dy) - c.r;
}

static float sdf_box_eval(float px, float py, const ObstacleBox& b) {
    float dx = fabsf(px - b.cx) - b.hw;
    float dy = fabsf(py - b.cy) - b.hh;
    float ox = dx > 0.0f ? dx : 0.0f;
    float oy = dy > 0.0f ? dy : 0.0f;
    return sqrtf(ox * ox + oy * oy) + std::min(std::max(dx, dy), 0.0f);
}

void bake_sdf(const SceneDesc& scene, float world_half, int res,
              std::vector<float>& out) {
    out.resize(res * res);
    if (scene.obstacles.empty()) {
        std::fill(out.begin(), out.end(), 1.0e6f);
        return;
    }
    float step = 2.0f * world_half / static_cast<float>(res);
    for (int row = 0; row < res; ++row) {
        float wy = -world_half + (row + 0.5f) * step;
        for (int col = 0; col < res; ++col) {
            float wx = -world_half + (col + 0.5f) * step;
            float d = 1.0e6f;
            for (const Obstacle& obs : scene.obstacles) {
                float od = (obs.type == ObstacleType::Circle)
                    ? sdf_circle_eval(wx, wy, obs.circle)
                    : sdf_box_eval(wx, wy, obs.box);
                d = std::min(d, od);
            }
            out[row * res + col] = d;
        }
    }
}

void seed_from_scene_desc(const SceneDesc& scene, std::vector<float2>& out) {
    float sp  = scene.spacing;
    float jit = sp * 0.15f;
    out.clear();
    for (const SpawnRect& sr : scene.spawn_rects) {
        int cols = std::max(1, static_cast<int>(sr.width  / sp));
        int rows = std::max(1, static_cast<int>(sr.height / sp));
        for (int row = 0; row < rows; ++row) {
            float off_x = (row & 1) ? jit : 0.0f;
            for (int col = 0; col < cols; ++col) {
                out.push_back({sr.x + col * sp + off_x, sr.y + row * sp});
            }
        }
    }
}

bool load_scene_json(const std::string& path, SceneDesc& out, std::string& error) {
    std::ifstream f(path);
    if (!f.is_open()) { error = "cannot open: " + path; return false; }
    try {
        json j = json::parse(f);
        out = SceneDesc{};
        out.name    = j.value("name", fs::path(path).stem().string());
        out.spacing = j.value("spacing", 0.019f);
        for (const auto& sr : j.value("spawn_rects", json::array())) {
            out.spawn_rects.push_back({sr.value("x", 0.0f), sr.value("y", 0.0f),
                                       sr.value("width", 1.0f), sr.value("height", 1.0f)});
        }
        for (const auto& ob : j.value("obstacles", json::array())) {
            Obstacle obs{};
            std::string type = ob.value("type", "circle");
            if (type == "circle") {
                obs.type = ObstacleType::Circle;
                obs.circle = {ob.value("cx", 0.0f), ob.value("cy", 0.0f), ob.value("r", 0.5f)};
            } else if (type == "box") {
                obs.type = ObstacleType::Box;
                // JSON: x/y = top-left corner (min-x, max-y), w/h = size
                float x = ob.value("x", 0.0f);
                float y = ob.value("y", 0.0f);
                float w = ob.value("w", 1.0f);
                float h = ob.value("h", 1.0f);
                obs.box.cx = x + w * 0.5f;
                obs.box.cy = y - h * 0.5f;
                obs.box.hw = w * 0.5f;
                obs.box.hh = h * 0.5f;
            } else {
                error = "unknown obstacle type: " + type;
                return false;
            }
            out.obstacles.push_back(obs);
        }
        return true;
    } catch (const json::exception& e) {
        error = std::string("JSON error: ") + e.what();
        return false;
    }
}

std::vector<std::string> list_scene_files(const std::string& dir) {
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (entry.path().extension() == ".json")
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}
