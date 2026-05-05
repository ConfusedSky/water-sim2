#include "kernel.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

constexpr int kParticlesX = 32;
constexpr int kParticlesY = 64;
constexpr int kParticleCount = kParticlesX * kParticlesY;
constexpr int kMaxNeighbors = 64;
constexpr int kThreadsPerBlock = 128;
constexpr int kSolverIterations = 4;
constexpr float kPi = 3.14159265358979323846f;
constexpr int kMouseButtonNone = 0;
constexpr int kMouseButtonDrag = 1;
constexpr int kMouseButtonPush = 2;

struct SimParams {
    float2 box_min;
    float2 box_max;
    float gravity;
    float particle_radius;
    float kernel_radius;
    float rest_density;
    float lambda_epsilon;
    float tensile_k;
    float tensile_n;
    float tensile_q;
    float velocity_damping;
    float boundary_bounce;
    float max_position_correction;
    float max_speed;
    int solver_iterations;
};

struct DeviceStats {
    float density_sum;
    unsigned int density_max_bits;
    float speed_sum;
};

__constant__ SimParams c_params;

float2* g_pos = nullptr;
float2* g_vel = nullptr;
float2* g_predicted = nullptr;
float2* g_delta_p = nullptr;
float* g_lambda = nullptr;
float* g_density = nullptr;
int* g_neighbors = nullptr;
int* g_neighbor_counts = nullptr;
DeviceStats* g_stats = nullptr;

bool g_initialized = false;
SimulationStats g_last_stats{};
std::vector<float2> g_initial_positions;

__host__ __device__ inline float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline float2 operator*(float2 v, float s) {
    return make_float2(v.x * s, v.y * s);
}

__host__ __device__ inline float2 operator/(float2 v, float s) {
    return make_float2(v.x / s, v.y / s);
}

__device__ inline float dot2(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ inline float length2(float2 v) {
    return dot2(v, v);
}

__device__ inline float length1(float2 v) {
    return sqrtf(length2(v));
}

__host__ __device__ inline float2 clamp_length(float2 v, float max_length) {
    float len2 = v.x * v.x + v.y * v.y;
    if (len2 <= max_length * max_length || len2 <= 1.0e-12f) {
        return v;
    }
    float scale = max_length / sqrtf(len2);
    return v * scale;
}

__device__ inline float poly6_weight(float2 delta) {
    float h = c_params.kernel_radius;
    float h2 = h * h;
    float r2 = length2(delta);
    if (r2 >= h2) {
        return 0.0f;
    }
    float term = h2 - r2;
    float coeff = 4.0f / (kPi * h2 * h2 * h2 * h2);
    return coeff * term * term * term;
}

__device__ inline float2 spiky_gradient(float2 delta) {
    float h = c_params.kernel_radius;
    float r = length1(delta);
    if (r >= h || r <= 1.0e-6f) {
        return make_float2(0.0f, 0.0f);
    }
    float coeff = -30.0f / (kPi * h * h * h * h * h);
    float scale = coeff * (h - r) * (h - r) / r;
    return delta * scale;
}

__device__ inline float compute_density_at(const float2* positions, int i) {
    float density = poly6_weight(make_float2(0.0f, 0.0f));
    float2 pi = positions[i];
    for (int j = 0; j < kParticleCount; ++j) {
        if (i == j) {
            continue;
        }
        density += poly6_weight(pi - positions[j]);
    }
    return density;
}

__global__ void apply_external_forces_kernel(float2* vel, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    vel[i].y += c_params.gravity * dt;
}

__global__ void apply_mouse_force_kernel(const float2* pos, float2* vel, int n,
                                         float dt, MouseState mouse) {
    if (mouse.button_state == kMouseButtonNone || mouse.radius <= 1.0e-6f ||
        mouse.strength <= 0.0f) {
        return;
    }

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float2 delta = pos[i] - mouse.pos;
    float dist = length1(delta);
    if (dist >= mouse.radius) {
        return;
    }

    float falloff = 1.0f - dist / mouse.radius;
    float2 impulse = make_float2(0.0f, 0.0f);

    if (mouse.button_state == kMouseButtonDrag) {
        impulse = mouse.vel * (mouse.strength * falloff * dt);
    } else if (mouse.button_state == kMouseButtonPush && dist > 1.0e-5f) {
        float2 radial = delta / dist;
        impulse = radial * (mouse.strength * falloff * dt);
    }

    vel[i] = vel[i] + impulse;
}

__global__ void predict_positions_kernel(const float2* pos, const float2* vel,
                                         float2* predicted, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    predicted[i] = pos[i] + vel[i] * dt;
}

__global__ void find_neighbors_naive_kernel(const float2* positions,
                                            int* neighbors,
                                            int* neighbor_counts, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    int count = 0;
    float2 pi = positions[i];
    float h2 = c_params.kernel_radius * c_params.kernel_radius;

    for (int j = 0; j < n; ++j) {
        if (i == j) {
            continue;
        }
        float2 delta = pi - positions[j];
        if (length2(delta) >= h2) {
            continue;
        }
        if (count < kMaxNeighbors) {
            neighbors[i * kMaxNeighbors + count] = j;
            ++count;
        }
    }

    neighbor_counts[i] = count;
}

__global__ void compute_lambda_kernel(const float2* positions,
                                      const int* neighbors,
                                      const int* neighbor_counts,
                                      float* lambda, float* density, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float2 pi = positions[i];
    float rho = poly6_weight(make_float2(0.0f, 0.0f));
    float2 grad_i = make_float2(0.0f, 0.0f);
    float sum_grad2 = 0.0f;
    int count = neighbor_counts[i];

    for (int k = 0; k < count; ++k) {
        int j = neighbors[i * kMaxNeighbors + k];
        float2 grad_w = spiky_gradient(pi - positions[j]);
        float2 grad_j = grad_w / c_params.rest_density;
        rho += poly6_weight(pi - positions[j]);
        sum_grad2 += dot2(grad_j, grad_j);
        grad_i = grad_i + grad_w / c_params.rest_density;
    }

    sum_grad2 += dot2(grad_i, grad_i);

    float constraint = rho / c_params.rest_density - 1.0f;
    lambda[i] = -constraint / (sum_grad2 + c_params.lambda_epsilon);
    density[i] = rho;
}

__global__ void compute_delta_p_kernel(const float2* positions,
                                       const int* neighbors,
                                       const int* neighbor_counts,
                                       const float* lambda,
                                       float2* delta_p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float2 pi = positions[i];
    float2 delta = make_float2(0.0f, 0.0f);
    float q_ref = poly6_weight(make_float2(c_params.kernel_radius * c_params.tensile_q,
                                           0.0f));
    int count = neighbor_counts[i];

    for (int k = 0; k < count; ++k) {
        int j = neighbors[i * kMaxNeighbors + k];
        float2 rij = pi - positions[j];
        float w = poly6_weight(rij);
        float s_corr = 0.0f;
        if (q_ref > 1.0e-6f) {
            float ratio = w / q_ref;
            s_corr = -c_params.tensile_k * powf(ratio, c_params.tensile_n);
        }
        delta = delta + spiky_gradient(rij) * (lambda[i] + lambda[j] + s_corr);
    }

    delta_p[i] = clamp_length(delta / c_params.rest_density,
                              c_params.max_position_correction);
}

__global__ void apply_delta_p_kernel(float2* predicted, const float2* delta_p,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    predicted[i] = predicted[i] + delta_p[i];
}

__global__ void update_velocity_and_position_kernel(float2* pos, float2* vel,
                                                    const float2* predicted,
                                                    int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float2 next_vel = (predicted[i] - pos[i]) / dt;
    next_vel = clamp_length(next_vel, c_params.max_speed);
    vel[i] = next_vel * c_params.velocity_damping;
    pos[i] = predicted[i];
}

__global__ void enforce_box_boundary_kernel(float2* pos, float2* vel, int n,
                                            int reflect_velocity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float2 p = pos[i];
    float2 v = vel[i];
    float min_x = c_params.box_min.x + c_params.particle_radius;
    float max_x = c_params.box_max.x - c_params.particle_radius;
    float min_y = c_params.box_min.y + c_params.particle_radius;
    float max_y = c_params.box_max.y - c_params.particle_radius;

    if (p.x < min_x) {
        p.x = min_x;
        if (reflect_velocity && v.x < 0.0f) {
            v.x *= -c_params.boundary_bounce;
        }
    } else if (p.x > max_x) {
        p.x = max_x;
        if (reflect_velocity && v.x > 0.0f) {
            v.x *= -c_params.boundary_bounce;
        }
    }

    if (p.y < min_y) {
        p.y = min_y;
        if (reflect_velocity && v.y < 0.0f) {
            v.y *= -c_params.boundary_bounce;
        }
    } else if (p.y > max_y) {
        p.y = max_y;
        if (reflect_velocity && v.y > 0.0f) {
            v.y *= -c_params.boundary_bounce;
        }
    }

    pos[i] = p;
    vel[i] = v;
}

__global__ void compute_density_only_kernel(const float2* positions,
                                            float* density, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    density[i] = compute_density_at(positions, i);
}

__global__ void gather_stats_kernel(const float* density, const float2* vel,
                                    DeviceStats* stats, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float rho = density[i];
    float speed = length1(vel[i]);
    atomicAdd(&stats->density_sum, rho);
    atomicAdd(&stats->speed_sum, speed);
    atomicMax(&stats->density_max_bits, __float_as_uint(rho));
}

__global__ void export_render_particles_kernel(const float2* pos,
                                               const float* density,
                                               float4* render_particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    render_particles[i] =
        make_float4(pos[i].x, pos[i].y, density[i] / c_params.rest_density, 1.0f);
}

void allocate_simulation_buffers() {
    CUDA_CHECK(cudaMalloc(&g_pos, kParticleCount * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&g_vel, kParticleCount * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&g_predicted, kParticleCount * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&g_delta_p, kParticleCount * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&g_lambda, kParticleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_density, kParticleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_neighbors, kParticleCount * kMaxNeighbors * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_neighbor_counts, kParticleCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g_stats, sizeof(DeviceStats)));
}

void seed_initial_positions() {
    g_initial_positions.resize(kParticleCount);

    constexpr float spacing = 0.019f;
    constexpr float start_x = -0.80f;
    constexpr float start_y = -0.35f;

    for (int y = 0; y < kParticlesY; ++y) {
        for (int x = 0; x < kParticlesX; ++x) {
            int i = y * kParticlesX + x;
            float offset_x = (y & 1) ? spacing * 0.15f : 0.0f;
            g_initial_positions[i] =
                make_float2(start_x + x * spacing + offset_x, start_y + y * spacing);
        }
    }
}

float host_poly6_weight(float2 delta, float kernel_radius) {
    float h2 = kernel_radius * kernel_radius;
    float r2 = delta.x * delta.x + delta.y * delta.y;
    if (r2 >= h2) {
        return 0.0f;
    }
    float term = h2 - r2;
    float coeff = 4.0f / (kPi * h2 * h2 * h2 * h2);
    return coeff * term * term * term;
}

float estimate_rest_density(float kernel_radius) {
    std::vector<float> densities(kParticleCount, 0.0f);
    for (int i = 0; i < kParticleCount; ++i) {
        float rho = host_poly6_weight(make_float2(0.0f, 0.0f), kernel_radius);
        for (int j = 0; j < kParticleCount; ++j) {
            if (i == j) {
                continue;
            }
            rho += host_poly6_weight(g_initial_positions[i] - g_initial_positions[j],
                                     kernel_radius);
        }
        densities[i] = rho;
    }

    std::sort(densities.begin(), densities.end());
    int start = (kParticleCount * 3) / 4;
    float sum = 0.0f;
    for (int i = start; i < kParticleCount; ++i) {
        sum += densities[i];
    }
    return sum / static_cast<float>(kParticleCount - start);
}

void upload_params() {
    SimParams params{};
    params.box_min = make_float2(-0.88f, -0.88f);
    params.box_max = make_float2(0.88f, 0.88f);
    params.gravity = -5.5f;
    params.particle_radius = 0.0075f;
    params.kernel_radius = 0.05f;
    params.rest_density = estimate_rest_density(params.kernel_radius) * 0.95f;
    params.lambda_epsilon = 400.0f;
    params.tensile_k = 0.0003f;
    params.tensile_n = 4.0f;
    params.tensile_q = 0.3f;
    params.velocity_damping = 0.992f;
    params.boundary_bounce = 0.02f;
    params.max_position_correction = params.particle_radius * 0.75f;
    params.max_speed = 2.75f;
    params.solver_iterations = kSolverIterations;
    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &params, sizeof(params)));
}

void launch_density_and_stats_passes(int blocks) {
    compute_density_only_kernel<<<blocks, kThreadsPerBlock>>>(g_pos, g_density,
                                                              kParticleCount);
    CUDA_CHECK(cudaMemset(g_stats, 0, sizeof(DeviceStats)));
    gather_stats_kernel<<<blocks, kThreadsPerBlock>>>(g_density, g_vel, g_stats,
                                                      kParticleCount);
}

float bits_to_float(unsigned int bits) {
    union {
        unsigned int u;
        float f;
    } value{};
    value.u = bits;
    return value.f;
}

}  // namespace

void init_simulation() {
    if (g_initialized) {
        return;
    }

    seed_initial_positions();
    upload_params();
    allocate_simulation_buffers();
    reset_simulation();
    g_initialized = true;
}

void reset_simulation() {
    if (!g_pos) {
        return;
    }

    std::vector<float2> zero_vel(kParticleCount, make_float2(0.0f, 0.0f));
    CUDA_CHECK(cudaMemcpy(g_pos, g_initial_positions.data(),
                          kParticleCount * sizeof(float2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_predicted, g_initial_positions.data(),
                          kParticleCount * sizeof(float2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_vel, zero_vel.data(), kParticleCount * sizeof(float2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g_delta_p, 0, kParticleCount * sizeof(float2)));
    CUDA_CHECK(cudaMemset(g_lambda, 0, kParticleCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_density, 0, kParticleCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_neighbor_counts, 0, kParticleCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(g_neighbors, 0, kParticleCount * kMaxNeighbors * sizeof(int)));
    CUDA_CHECK(cudaMemset(g_stats, 0, sizeof(DeviceStats)));
    g_last_stats = {};
}

void shutdown_simulation() {
    if (!g_initialized) {
        return;
    }

    CUDA_CHECK(cudaFree(g_pos));
    CUDA_CHECK(cudaFree(g_vel));
    CUDA_CHECK(cudaFree(g_predicted));
    CUDA_CHECK(cudaFree(g_delta_p));
    CUDA_CHECK(cudaFree(g_lambda));
    CUDA_CHECK(cudaFree(g_density));
    CUDA_CHECK(cudaFree(g_neighbors));
    CUDA_CHECK(cudaFree(g_neighbor_counts));
    CUDA_CHECK(cudaFree(g_stats));

    g_pos = nullptr;
    g_vel = nullptr;
    g_predicted = nullptr;
    g_delta_p = nullptr;
    g_lambda = nullptr;
    g_density = nullptr;
    g_neighbors = nullptr;
    g_neighbor_counts = nullptr;
    g_stats = nullptr;
    g_initial_positions.clear();
    g_initialized = false;
}

int get_particle_count() {
    return kParticleCount;
}

int get_solver_iterations() {
    return kSolverIterations;
}

SimulationStats get_simulation_stats() {
    if (!g_initialized) {
        return {};
    }

    DeviceStats device_stats{};
    CUDA_CHECK(cudaMemcpy(&device_stats, g_stats, sizeof(device_stats),
                          cudaMemcpyDeviceToHost));
    g_last_stats.avg_density = device_stats.density_sum / kParticleCount;
    g_last_stats.max_density = bits_to_float(device_stats.density_max_bits);
    g_last_stats.avg_speed = device_stats.speed_sum / kParticleCount;
    return g_last_stats;
}

void step_simulation(float dt, const MouseState& mouse, float4* render_particles) {
    if (!g_initialized) {
        init_simulation();
    }

    int blocks = (kParticleCount + kThreadsPerBlock - 1) / kThreadsPerBlock;

    apply_external_forces_kernel<<<blocks, kThreadsPerBlock>>>(g_vel, kParticleCount,
                                                               dt);
    apply_mouse_force_kernel<<<blocks, kThreadsPerBlock>>>(g_pos, g_vel,
                                                           kParticleCount, dt,
                                                           mouse);
    predict_positions_kernel<<<blocks, kThreadsPerBlock>>>(g_pos, g_vel, g_predicted,
                                                           kParticleCount, dt);
    enforce_box_boundary_kernel<<<blocks, kThreadsPerBlock>>>(g_predicted, g_vel,
                                                              kParticleCount, 0);

    for (int iter = 0; iter < kSolverIterations; ++iter) {
        find_neighbors_naive_kernel<<<blocks, kThreadsPerBlock>>>(
            g_predicted, g_neighbors, g_neighbor_counts, kParticleCount);
        compute_lambda_kernel<<<blocks, kThreadsPerBlock>>>(
            g_predicted, g_neighbors, g_neighbor_counts, g_lambda, g_density,
            kParticleCount);
        compute_delta_p_kernel<<<blocks, kThreadsPerBlock>>>(
            g_predicted, g_neighbors, g_neighbor_counts, g_lambda, g_delta_p,
            kParticleCount);
        apply_delta_p_kernel<<<blocks, kThreadsPerBlock>>>(g_predicted, g_delta_p,
                                                           kParticleCount);
        enforce_box_boundary_kernel<<<blocks, kThreadsPerBlock>>>(
            g_predicted, g_vel, kParticleCount, 0);
    }

    update_velocity_and_position_kernel<<<blocks, kThreadsPerBlock>>>(
        g_pos, g_vel, g_predicted, kParticleCount, dt);
    enforce_box_boundary_kernel<<<blocks, kThreadsPerBlock>>>(g_pos, g_vel,
                                                              kParticleCount, 1);

    launch_density_and_stats_passes(blocks);
    export_render_particles_kernel<<<blocks, kThreadsPerBlock>>>(
        g_pos, g_density, render_particles, kParticleCount);
    CUDA_CHECK(cudaGetLastError());
}
