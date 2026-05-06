#pragma once

#include <vector>
#include <vector_types.h>

#include "scenes.h"

struct SimulationStats {
    float avg_density;
    float max_density;
    float avg_speed;
    int   grid_h;        // current adaptive Y cell count
    int   grid_h_max;    // maximum possible Y cell count (full domain)
};

struct MouseState {
    float3 ray_origin;
    float3 ray_dir;
    float  radius;
    float  strength;
    int    button_state;
};

struct TunableParams {
    float rest_density;
    float kernel_radius;
    int   solver_iterations;
    float lambda_epsilon;
    float tensile_k;
    float tensile_n;
    float tensile_q;
    float gravity;
    float velocity_damping;
    float boundary_bounce;
    float viscosity_c;
    float vorticity_eps;
    float max_speed;
    float max_position_correction;
};

void init_simulation();
void reset_simulation();
void shutdown_simulation();

int              get_particle_count();
int              get_solver_iterations();
SimulationStats  get_simulation_stats();

TunableParams get_tunable_params();
void          set_tunable_params(const TunableParams& params);

SceneId get_active_scene();
void    set_active_scene(SceneId id);

void step_simulation(float dt, const MouseState& mouse, float4* render_particles);
