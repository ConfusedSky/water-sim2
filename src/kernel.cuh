#pragma once

#include <vector_types.h>

struct SimulationStats {
    float avg_density;
    float max_density;
    float avg_speed;
};

void init_simulation();
void reset_simulation();
void shutdown_simulation();

int get_particle_count();
int get_solver_iterations();
SimulationStats get_simulation_stats();

void step_simulation(float dt, float4* render_particles);
