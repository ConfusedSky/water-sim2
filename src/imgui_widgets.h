#pragma once

#include <imgui.h>

#include <cmath>

namespace ui {

// Drag-to-change float input. Drag speed scales with |*v| so the increment
// stays proportional to the current magnitude. Open-ended (no clamping).
// Double-click on the widget for keyboard text entry (ImGui built-in).
inline bool ProportionalDragFloat(const char* label, float* v,
                                  const char* format = "%.3f",
                                  float rel_speed = 0.01f,
                                  float min_speed = 1e-6f) {
    float speed = std::fabs(*v) * rel_speed;
    if (speed < min_speed) speed = min_speed;
    return ImGui::DragFloat(label, v, speed, 0.0f, 0.0f, format);
}

}  // namespace ui
