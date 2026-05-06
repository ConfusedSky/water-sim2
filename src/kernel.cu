#include "kernel.cuh"

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr int   kParticleCount    = kSceneParticleCount;
constexpr int   kMaxNeighbors     = 96;
constexpr int   kThreadsPerBlock  = 128;
constexpr float kPi               = 3.14159265358979323846f;
constexpr int   kMouseButtonNone  = 0;
constexpr int   kMouseButtonPush  = 2;
constexpr float kMinCellSize      = 0.05f;

// ---------------------------------------------------------------------------
// float3 operators
// ---------------------------------------------------------------------------

__host__ __device__ inline float3 operator+(float3 a, float3 b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__host__ __device__ inline float3 operator-(float3 a, float3 b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__host__ __device__ inline float3 operator*(float3 v, float s) {
  return make_float3(v.x*s, v.y*s, v.z*s);
}
__host__ __device__ inline float3 operator/(float3 v, float s) {
  return make_float3(v.x/s, v.y/s, v.z/s);
}
__host__ __device__ inline float3& operator+=(float3& a, float3 b) {
  a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
__host__ __device__ inline float dot3(float3 a, float3 b) {
  return a.x*b.x + a.y*b.y + a.z*b.z;
}
__host__ __device__ inline float length2(float3 v) { return dot3(v, v); }
__host__ __device__ inline float length1(float3 v) { return sqrtf(length2(v)); }
__device__ inline float3 cross3(float3 a, float3 b) {
  return make_float3(a.y*b.z - a.z*b.y,
                     a.z*b.x - a.x*b.z,
                     a.x*b.y - a.y*b.x);
}
__host__ __device__ inline float3 clamp_length(float3 v, float max_len) {
  float len2 = length2(v);
  if (len2 <= max_len * max_len || len2 <= 1.0e-12f) return v;
  return v * (max_len / sqrtf(len2));
}

// ---------------------------------------------------------------------------
// Sim params
// ---------------------------------------------------------------------------

struct SimParams {
  float3 box_min;
  float3 box_max;
  float  gravity;
  float  particle_radius;
  float  kernel_radius;
  float  rest_density;
  float  lambda_epsilon;
  float  tensile_k;
  float  tensile_n;
  float  tensile_q;
  float  velocity_damping;
  float  boundary_bounce;
  float  max_position_correction;
  float  max_speed;
  float  viscosity_c;
  float  vorticity_eps;
  int    solver_iterations;
  float  cell_size;
  int    grid_w;
  int    grid_h;
  int    grid_d;
};

struct DeviceStats {
  float        density_sum;
  unsigned int density_max_bits;
  float        speed_sum;
};

__constant__ SimParams c_params;
SimParams g_params{};

float3 *g_pos           = nullptr;
float3 *g_vel           = nullptr;
float3 *g_vel_scratch   = nullptr;
float3 *g_predicted     = nullptr;
float3 *g_delta_p       = nullptr;
float3 *g_vorticity     = nullptr;
float  *g_lambda        = nullptr;
float  *g_density       = nullptr;
int    *g_neighbors     = nullptr;
int    *g_neighbor_counts = nullptr;
int    *g_cell_hash     = nullptr;
int    *g_particle_index = nullptr;
int    *g_cell_start    = nullptr;
int    *g_cell_end      = nullptr;
int     g_max_cells     = 0;
DeviceStats *g_stats    = nullptr;

bool g_initialized = false;
SimulationStats g_last_stats{};
std::vector<float3> g_initial_positions;
SceneId g_active_scene = SceneId::CubeFull;

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------

__device__ inline int3 cell_coords_for(float3 p) {
  int cx = static_cast<int>(floorf((p.x - c_params.box_min.x) / c_params.cell_size));
  int cy = static_cast<int>(floorf((p.y - c_params.box_min.y) / c_params.cell_size));
  int cz = static_cast<int>(floorf((p.z - c_params.box_min.z) / c_params.cell_size));
  cx = max(0, min(cx, c_params.grid_w - 1));
  cy = max(0, min(cy, c_params.grid_h - 1));
  cz = max(0, min(cz, c_params.grid_d - 1));
  return make_int3(cx, cy, cz);
}

__device__ inline int cell_index_for(int3 c) {
  return c.z * c_params.grid_w * c_params.grid_h + c.y * c_params.grid_w + c.x;
}

// ---------------------------------------------------------------------------
// SPH kernels (3-D)
// ---------------------------------------------------------------------------

__device__ inline float poly6_weight(float3 delta) {
  float h  = c_params.kernel_radius;
  float h2 = h * h;
  float r2 = length2(delta);
  if (r2 >= h2) return 0.0f;
  float term  = h2 - r2;
  // 315 / (64 π h^9)
  float coeff = 315.0f / (64.0f * kPi * h2 * h2 * h2 * h2 * h);
  return coeff * term * term * term;
}

__device__ inline float3 spiky_gradient(float3 delta) {
  float h = c_params.kernel_radius;
  float r = length1(delta);
  if (r >= h || r <= 1.0e-6f) return make_float3(0.0f, 0.0f, 0.0f);
  // -45 / (π h^6) * (h-r)^2 / r * delta
  float coeff = -45.0f / (kPi * h * h * h * h * h * h);
  float scale = coeff * (h - r) * (h - r) / r;
  return delta * scale;
}

// ---------------------------------------------------------------------------
// Simulation kernels
// ---------------------------------------------------------------------------

__global__ void apply_external_forces_kernel(float3 *vel, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  vel[i].y += c_params.gravity * dt;
}

__global__ void apply_mouse_force_kernel(const float3 *pos, float3 *vel,
                                          int n, float dt, MouseState mouse) {
  if (mouse.button_state == kMouseButtonNone) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // Distance from particle to ray
  float3 to_p   = pos[i] - mouse.ray_origin;
  float  t      = dot3(to_p, mouse.ray_dir);
  float3 closest = mouse.ray_origin + mouse.ray_dir * t;
  float3 delta  = pos[i] - closest;
  float  dist   = length1(delta);
  if (dist >= mouse.radius) return;

  float falloff = 1.0f - dist / mouse.radius;
  if (mouse.button_state == kMouseButtonPush && dist > 1.0e-5f) {
    float3 radial = delta / dist;
    vel[i] += radial * (mouse.strength * falloff * dt);
  }
}

__global__ void predict_positions_kernel(const float3 *pos, const float3 *vel,
                                          float3 *predicted, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  predicted[i] = pos[i] + vel[i] * dt;
}

__global__ void compute_cell_hash_kernel(const float3 *positions,
                                          int *cell_hash, int *particle_index,
                                          int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  cell_hash[i]      = cell_index_for(cell_coords_for(positions[i]));
  particle_index[i] = i;
}

__global__ void find_cell_starts_kernel(const int *cell_hash_sorted,
                                         int *cell_start, int *cell_end, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int hi = cell_hash_sorted[i];
  if (i == 0) {
    cell_start[hi] = 0;
  } else {
    int hp = cell_hash_sorted[i - 1];
    if (hp != hi) { cell_start[hi] = i; cell_end[hp] = i; }
  }
  if (i == n - 1) cell_end[hi] = n;
}

__global__ void find_neighbors_grid_kernel(const float3 *positions,
                                            const int *particle_index,
                                            const int *cell_start,
                                            const int *cell_end,
                                            int *neighbors, int *neighbor_counts,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 pi  = positions[i];
  int3   c   = cell_coords_for(pi);
  float  h2  = c_params.kernel_radius * c_params.kernel_radius;
  int    cnt = 0;

  for (int dz = -1; dz <= 1; ++dz) {
    int nz = c.z + dz;
    if (nz < 0 || nz >= c_params.grid_d) continue;
    for (int dy = -1; dy <= 1; ++dy) {
      int ny = c.y + dy;
      if (ny < 0 || ny >= c_params.grid_h) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = c.x + dx;
        if (nx < 0 || nx >= c_params.grid_w) continue;
        int hash  = nz * c_params.grid_w * c_params.grid_h + ny * c_params.grid_w + nx;
        int start = cell_start[hash];
        int end   = cell_end[hash];
        for (int idx = start; idx < end; ++idx) {
          int j = particle_index[idx];
          if (j == i) continue;
          if (length2(pi - positions[j]) >= h2) continue;
          if (cnt < kMaxNeighbors) neighbors[i * kMaxNeighbors + cnt++] = j;
        }
      }
    }
  }
  neighbor_counts[i] = cnt;
}

__global__ void compute_density_grid_kernel(const float3 *positions,
                                             const int *particle_index,
                                             const int *cell_start,
                                             const int *cell_end,
                                             float *density, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 pi  = positions[i];
  int3   c   = cell_coords_for(pi);
  float  rho = poly6_weight(make_float3(0.0f, 0.0f, 0.0f));

  for (int dz = -1; dz <= 1; ++dz) {
    int nz = c.z + dz;
    if (nz < 0 || nz >= c_params.grid_d) continue;
    for (int dy = -1; dy <= 1; ++dy) {
      int ny = c.y + dy;
      if (ny < 0 || ny >= c_params.grid_h) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = c.x + dx;
        if (nx < 0 || nx >= c_params.grid_w) continue;
        int hash  = nz * c_params.grid_w * c_params.grid_h + ny * c_params.grid_w + nx;
        int start = cell_start[hash];
        int end   = cell_end[hash];
        for (int idx = start; idx < end; ++idx) {
          int j = particle_index[idx];
          if (j == i) continue;
          rho += poly6_weight(pi - positions[j]);
        }
      }
    }
  }
  density[i] = rho;
}

__global__ void compute_lambda_kernel(const float3 *positions,
                                       const int *neighbors,
                                       const int *neighbor_counts,
                                       float *lambda, float *density, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 pi     = positions[i];
  float  rho    = poly6_weight(make_float3(0.0f, 0.0f, 0.0f));
  float3 grad_i = make_float3(0.0f, 0.0f, 0.0f);
  float  sum_g2 = 0.0f;
  int    cnt    = neighbor_counts[i];

  for (int k = 0; k < cnt; ++k) {
    int    j      = neighbors[i * kMaxNeighbors + k];
    float3 gw     = spiky_gradient(pi - positions[j]);
    float3 grad_j = gw / c_params.rest_density;
    rho   += poly6_weight(pi - positions[j]);
    sum_g2 += dot3(grad_j, grad_j);
    grad_i += gw / c_params.rest_density;
  }
  sum_g2 += dot3(grad_i, grad_i);

  float constraint = rho / c_params.rest_density - 1.0f;
  lambda[i]  = -constraint / (sum_g2 + c_params.lambda_epsilon);
  density[i] = rho;
}

__global__ void compute_delta_p_kernel(const float3 *positions,
                                        const int *neighbors,
                                        const int *neighbor_counts,
                                        const float *lambda,
                                        float3 *delta_p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 pi    = positions[i];
  float3 delta = make_float3(0.0f, 0.0f, 0.0f);
  float  q_ref = poly6_weight(
      make_float3(c_params.kernel_radius * c_params.tensile_q, 0.0f, 0.0f));
  int cnt = neighbor_counts[i];

  for (int k = 0; k < cnt; ++k) {
    int    j   = neighbors[i * kMaxNeighbors + k];
    float3 rij = pi - positions[j];
    float  w   = poly6_weight(rij);
    float  s_corr = 0.0f;
    if (q_ref > 1.0e-6f) {
      float ratio = w / q_ref;
      s_corr = -c_params.tensile_k * powf(ratio, c_params.tensile_n);
    }
    delta += spiky_gradient(rij) * (lambda[i] + lambda[j] + s_corr);
  }
  delta_p[i] = clamp_length(delta / c_params.rest_density,
                             c_params.max_position_correction);
}

__global__ void apply_delta_p_kernel(float3 *predicted, const float3 *delta_p,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  predicted[i] = predicted[i] + delta_p[i];
}

__global__ void update_velocity_and_position_kernel(float3 *pos, float3 *vel,
                                                     const float3 *predicted,
                                                     int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float3 next_vel = (predicted[i] - pos[i]) / dt;
  next_vel = clamp_length(next_vel, c_params.max_speed);
  vel[i]   = next_vel * c_params.velocity_damping;
  pos[i]   = predicted[i];
}

__global__ void enforce_box_boundary_kernel(float3 *pos, float3 *vel,
                                             int n, int reflect) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 p = pos[i];
  float3 v = vel[i];
  float r  = c_params.particle_radius;

#define CLAMP_AXIS(axis, min_val, max_val)                          \
  if (p.axis < (min_val) + r) {                                     \
    p.axis = (min_val) + r;                                         \
    if (reflect && v.axis < 0.0f) v.axis *= -c_params.boundary_bounce; \
  } else if (p.axis > (max_val) - r) {                              \
    p.axis = (max_val) - r;                                         \
    if (reflect && v.axis > 0.0f) v.axis *= -c_params.boundary_bounce; \
  }

  CLAMP_AXIS(x, c_params.box_min.x, c_params.box_max.x)
  CLAMP_AXIS(y, c_params.box_min.y, c_params.box_max.y)
  CLAMP_AXIS(z, c_params.box_min.z, c_params.box_max.z)
#undef CLAMP_AXIS

  pos[i] = p;
  vel[i] = v;
}

__global__ void apply_xsph_viscosity_kernel(const float3 *positions,
                                             const float3 *vel_in,
                                             float3 *vel_out,
                                             const int *particle_index,
                                             const int *cell_start,
                                             const int *cell_end, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 pi    = positions[i];
  float3 vi    = vel_in[i];
  int3   c     = cell_coords_for(pi);
  float3 accum = make_float3(0.0f, 0.0f, 0.0f);

  for (int dz = -1; dz <= 1; ++dz) {
    int nz = c.z + dz;
    if (nz < 0 || nz >= c_params.grid_d) continue;
    for (int dy = -1; dy <= 1; ++dy) {
      int ny = c.y + dy;
      if (ny < 0 || ny >= c_params.grid_h) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = c.x + dx;
        if (nx < 0 || nx >= c_params.grid_w) continue;
        int hash  = nz * c_params.grid_w * c_params.grid_h + ny * c_params.grid_w + nx;
        int start = cell_start[hash];
        int end   = cell_end[hash];
        for (int idx = start; idx < end; ++idx) {
          int j = particle_index[idx];
          if (j == i) continue;
          float w = poly6_weight(pi - positions[j]);
          if (w <= 0.0f) continue;
          accum += (vel_in[j] - vi) * w;
        }
      }
    }
  }
  vel_out[i] = vi + accum * c_params.viscosity_c;
}

__global__ void compute_vorticity_kernel(const float3 *positions,
                                          const float3 *vel,
                                          float3 *vorticity,
                                          const int *particle_index,
                                          const int *cell_start,
                                          const int *cell_end, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 pi    = positions[i];
  float3 vi    = vel[i];
  int3   c     = cell_coords_for(pi);
  float3 omega = make_float3(0.0f, 0.0f, 0.0f);

  for (int dz = -1; dz <= 1; ++dz) {
    int nz = c.z + dz;
    if (nz < 0 || nz >= c_params.grid_d) continue;
    for (int dy = -1; dy <= 1; ++dy) {
      int ny = c.y + dy;
      if (ny < 0 || ny >= c_params.grid_h) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = c.x + dx;
        if (nx < 0 || nx >= c_params.grid_w) continue;
        int hash  = nz * c_params.grid_w * c_params.grid_h + ny * c_params.grid_w + nx;
        int start = cell_start[hash];
        int end   = cell_end[hash];
        for (int idx = start; idx < end; ++idx) {
          int j = particle_index[idx];
          if (j == i) continue;
          float3 grad = spiky_gradient(pi - positions[j]);
          if (length2(grad) < 1.0e-12f) continue;
          omega += cross3(vel[j] - vi, grad);
        }
      }
    }
  }
  vorticity[i] = omega;
}

__global__ void apply_vorticity_force_kernel(const float3 *positions,
                                              const float3 *vorticity,
                                              float3 *vel,
                                              const int *particle_index,
                                              const int *cell_start,
                                              const int *cell_end,
                                              int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float3 pi  = positions[i];
  int3   c   = cell_coords_for(pi);
  float3 eta = make_float3(0.0f, 0.0f, 0.0f);

  for (int dz = -1; dz <= 1; ++dz) {
    int nz = c.z + dz;
    if (nz < 0 || nz >= c_params.grid_d) continue;
    for (int dy = -1; dy <= 1; ++dy) {
      int ny = c.y + dy;
      if (ny < 0 || ny >= c_params.grid_h) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = c.x + dx;
        if (nx < 0 || nx >= c_params.grid_w) continue;
        int hash  = nz * c_params.grid_w * c_params.grid_h + ny * c_params.grid_w + nx;
        int start = cell_start[hash];
        int end   = cell_end[hash];
        for (int idx = start; idx < end; ++idx) {
          int j = particle_index[idx];
          if (j == i) continue;
          eta += spiky_gradient(pi - positions[j]) * length1(vorticity[j]);
        }
      }
    }
  }

  float eta_len2 = length2(eta);
  if (eta_len2 < 1.0e-10f) return;
  float3 N      = eta / sqrtf(eta_len2);
  float3 f_vort = cross3(N, vorticity[i]) * c_params.vorticity_eps;
  vel[i] += f_vort * dt;
}

__global__ void gather_stats_kernel(const float *density, const float3 *vel,
                                     DeviceStats *stats, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float rho   = density[i];
  float speed = length1(vel[i]);
  atomicAdd(&stats->density_sum, rho);
  atomicAdd(&stats->speed_sum, speed);
  atomicMax(&stats->density_max_bits, __float_as_uint(rho));
}

__global__ void export_render_particles_kernel(const float3 *pos,
                                                const float *density,
                                                float4 *render_particles,
                                                int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  render_particles[i] = make_float4(
      pos[i].x, pos[i].y, pos[i].z,
      density[i] / c_params.rest_density);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

void compute_grid_dims(SimParams &p) {
  float cs = std::max(p.kernel_radius, kMinCellSize);
  p.cell_size = cs;
  p.grid_w = std::max(1, static_cast<int>(std::ceil((p.box_max.x - p.box_min.x) / cs)));
  p.grid_h = std::max(1, static_cast<int>(std::ceil((p.box_max.y - p.box_min.y) / cs)));
  p.grid_d = std::max(1, static_cast<int>(std::ceil((p.box_max.z - p.box_min.z) / cs)));
}

void allocate_simulation_buffers() {
  CUDA_CHECK(cudaMalloc(&g_pos,           kParticleCount * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&g_vel,           kParticleCount * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&g_vel_scratch,   kParticleCount * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&g_predicted,     kParticleCount * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&g_delta_p,       kParticleCount * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&g_vorticity,     kParticleCount * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&g_lambda,        kParticleCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g_density,       kParticleCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&g_neighbors,     kParticleCount * kMaxNeighbors * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_neighbor_counts, kParticleCount * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_cell_hash,     kParticleCount * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_particle_index, kParticleCount * sizeof(int)));

  int mg = std::max(1, static_cast<int>(std::ceil(
      (g_params.box_max.x - g_params.box_min.x) / kMinCellSize)));
  g_max_cells = mg * mg * mg;
  CUDA_CHECK(cudaMalloc(&g_cell_start, g_max_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_cell_end,   g_max_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&g_stats,      sizeof(DeviceStats)));
}

float host_poly6_weight(float3 delta, float h) {
  float h2 = h * h;
  float r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
  if (r2 >= h2) return 0.0f;
  float term  = h2 - r2;
  float coeff = 315.0f / (64.0f * kPi * h2 * h2 * h2 * h2 * h);
  return coeff * term * term * term;
}

float estimate_rest_density(float kernel_radius) {
  constexpr int kMaxSamples = 2048;
  int sample_count = std::min(kParticleCount, kMaxSamples);
  int stride       = std::max(1, kParticleCount / sample_count);

  std::vector<float> densities;
  densities.reserve(sample_count);
  for (int i = 0; i < kParticleCount && (int)densities.size() < sample_count; i += stride) {
    float rho = host_poly6_weight(make_float3(0.0f, 0.0f, 0.0f), kernel_radius);
    for (int j = 0; j < kParticleCount; ++j) {
      if (i == j) continue;
      rho += host_poly6_weight(g_initial_positions[i] - g_initial_positions[j],
                               kernel_radius);
    }
    densities.push_back(rho);
  }

  std::sort(densities.begin(), densities.end());
  int taken = (int)densities.size();
  int start = (taken * 3) / 4;
  float sum = 0.0f;
  for (int i = start; i < taken; ++i) sum += densities[i];
  return sum / static_cast<float>(taken - start);
}

void upload_params() {
  g_params = SimParams{};
  float W = kWorldHalfExtent;
  g_params.box_min = make_float3(-W, -W, -W);
  g_params.box_max = make_float3( W,  W,  W);
  g_params.gravity        = -9.0f;
  g_params.particle_radius = 0.05f;
  g_params.kernel_radius   = 0.25f;
  g_params.rest_density    = estimate_rest_density(g_params.kernel_radius) * 0.95f;
  g_params.lambda_epsilon  = 300.0f;
  g_params.tensile_k       = 0.000001f;
  g_params.tensile_n       = 4.0f;
  g_params.tensile_q       = 0.2f;
  g_params.velocity_damping       = 0.9996f;
  g_params.boundary_bounce        = 0.2f;
  g_params.viscosity_c            = 0.000005f;
  g_params.vorticity_eps          = 0.00005f;
  g_params.max_position_correction = g_params.particle_radius * 0.75f;
  g_params.max_speed              = 100.0f;
  g_params.solver_iterations      = 5;
  compute_grid_dims(g_params);
  CUDA_CHECK(cudaMemcpyToSymbol(c_params, &g_params, sizeof(g_params)));
}

void rebuild_spatial_grid(const float3 *positions, int blocks, int num_cells) {
  compute_cell_hash_kernel<<<blocks, kThreadsPerBlock>>>(
      positions, g_cell_hash, g_particle_index, kParticleCount);
  thrust::sort_by_key(thrust::device, g_cell_hash, g_cell_hash + kParticleCount,
                      g_particle_index);
  CUDA_CHECK(cudaMemset(g_cell_start, 0, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMemset(g_cell_end,   0, num_cells * sizeof(int)));
  find_cell_starts_kernel<<<blocks, kThreadsPerBlock>>>(
      g_cell_hash, g_cell_start, g_cell_end, kParticleCount);
}

void run_post_step_passes(int blocks, float dt) {
  int num_cells = g_params.grid_w * g_params.grid_h * g_params.grid_d;
  rebuild_spatial_grid(g_pos, blocks, num_cells);

  if (g_params.vorticity_eps > 0.0f) {
    compute_vorticity_kernel<<<blocks, kThreadsPerBlock>>>(
        g_pos, g_vel, g_vorticity, g_particle_index, g_cell_start, g_cell_end,
        kParticleCount);
    apply_vorticity_force_kernel<<<blocks, kThreadsPerBlock>>>(
        g_pos, g_vorticity, g_vel, g_particle_index, g_cell_start, g_cell_end,
        kParticleCount, dt);
  }

  if (g_params.viscosity_c > 0.0f) {
    CUDA_CHECK(cudaMemcpyAsync(g_vel_scratch, g_vel, kParticleCount * sizeof(float3),
                               cudaMemcpyDeviceToDevice));
    apply_xsph_viscosity_kernel<<<blocks, kThreadsPerBlock>>>(
        g_pos, g_vel_scratch, g_vel, g_particle_index, g_cell_start, g_cell_end,
        kParticleCount);
  }

  compute_density_grid_kernel<<<blocks, kThreadsPerBlock>>>(
      g_pos, g_particle_index, g_cell_start, g_cell_end, g_density, kParticleCount);
  CUDA_CHECK(cudaMemset(g_stats, 0, sizeof(DeviceStats)));
  gather_stats_kernel<<<blocks, kThreadsPerBlock>>>(g_density, g_vel, g_stats,
                                                    kParticleCount);
}

float bits_to_float(unsigned int bits) {
  union { unsigned int u; float f; } v{};
  v.u = bits;
  return v.f;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void init_simulation() {
  if (g_initialized) return;
  seed_scene(g_active_scene, g_initial_positions);
  upload_params();
  allocate_simulation_buffers();
  reset_simulation();
  g_initialized = true;
}

void reset_simulation() {
  if (!g_pos) return;
  std::vector<float3> zero_vel(kParticleCount, make_float3(0.0f, 0.0f, 0.0f));
  CUDA_CHECK(cudaMemcpy(g_pos, g_initial_positions.data(),
                        kParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(g_predicted, g_initial_positions.data(),
                        kParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(g_vel, zero_vel.data(),
                        kParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(g_delta_p,        0, kParticleCount * sizeof(float3)));
  CUDA_CHECK(cudaMemset(g_vorticity,      0, kParticleCount * sizeof(float3)));
  CUDA_CHECK(cudaMemset(g_lambda,         0, kParticleCount * sizeof(float)));
  CUDA_CHECK(cudaMemset(g_density,        0, kParticleCount * sizeof(float)));
  CUDA_CHECK(cudaMemset(g_neighbor_counts,0, kParticleCount * sizeof(int)));
  CUDA_CHECK(cudaMemset(g_neighbors,      0, kParticleCount * kMaxNeighbors * sizeof(int)));
  CUDA_CHECK(cudaMemset(g_stats,          0, sizeof(DeviceStats)));
  g_last_stats = {};
}

void shutdown_simulation() {
  if (!g_initialized) return;
  cudaFree(g_pos);           cudaFree(g_vel);          cudaFree(g_vel_scratch);
  cudaFree(g_predicted);     cudaFree(g_delta_p);       cudaFree(g_vorticity);
  cudaFree(g_lambda);        cudaFree(g_density);
  cudaFree(g_neighbors);     cudaFree(g_neighbor_counts);
  cudaFree(g_cell_hash);     cudaFree(g_particle_index);
  cudaFree(g_cell_start);    cudaFree(g_cell_end);
  cudaFree(g_stats);
  g_pos = nullptr; g_vel = nullptr; g_vel_scratch = nullptr;
  g_predicted = nullptr; g_delta_p = nullptr; g_vorticity = nullptr;
  g_lambda = nullptr; g_density = nullptr;
  g_neighbors = nullptr; g_neighbor_counts = nullptr;
  g_cell_hash = nullptr; g_particle_index = nullptr;
  g_cell_start = nullptr; g_cell_end = nullptr;
  g_stats = nullptr; g_max_cells = 0;
  g_initial_positions.clear();
  g_initialized = false;
}

int get_particle_count()   { return kParticleCount; }
int get_solver_iterations(){ return g_params.solver_iterations; }
SceneId get_active_scene() { return g_active_scene; }

void set_active_scene(SceneId id) {
  g_active_scene = id;
  if (!g_initialized) return;
  seed_scene(g_active_scene, g_initial_positions);
  reset_simulation();
}

TunableParams get_tunable_params() {
  TunableParams t{};
  t.rest_density           = g_params.rest_density;
  t.kernel_radius          = g_params.kernel_radius;
  t.solver_iterations      = g_params.solver_iterations;
  t.lambda_epsilon         = g_params.lambda_epsilon;
  t.tensile_k              = g_params.tensile_k;
  t.tensile_n              = g_params.tensile_n;
  t.tensile_q              = g_params.tensile_q;
  t.gravity                = g_params.gravity;
  t.velocity_damping       = g_params.velocity_damping;
  t.boundary_bounce        = g_params.boundary_bounce;
  t.viscosity_c            = g_params.viscosity_c;
  t.vorticity_eps          = g_params.vorticity_eps;
  t.max_speed              = g_params.max_speed;
  t.max_position_correction = g_params.max_position_correction;
  return t;
}

void set_tunable_params(const TunableParams &p) {
  g_params.rest_density    = std::max(p.rest_density, 1.0e-3f);
  g_params.kernel_radius   = std::max(p.kernel_radius, 1.0e-4f);
  g_params.solver_iterations = std::max(1, std::min(p.solver_iterations, 32));
  g_params.lambda_epsilon  = std::max(p.lambda_epsilon, 0.0f);
  g_params.tensile_k       = std::max(p.tensile_k, 0.0f);
  g_params.tensile_n       = std::max(p.tensile_n, 0.0f);
  g_params.tensile_q       = std::max(p.tensile_q, 1.0e-4f);
  g_params.gravity         = p.gravity;
  g_params.velocity_damping       = std::max(0.0f, std::min(p.velocity_damping, 1.0f));
  g_params.boundary_bounce        = std::max(0.0f, std::min(p.boundary_bounce, 1.0f));
  g_params.viscosity_c            = std::max(p.viscosity_c, 0.0f);
  g_params.vorticity_eps          = std::max(p.vorticity_eps, 0.0f);
  g_params.max_speed              = std::max(p.max_speed, 1.0e-3f);
  g_params.max_position_correction = std::max(p.max_position_correction, 1.0e-5f);
  compute_grid_dims(g_params);
  CUDA_CHECK(cudaMemcpyToSymbol(c_params, &g_params, sizeof(g_params)));
}

SimulationStats get_simulation_stats() {
  if (!g_initialized) return {};
  DeviceStats ds{};
  CUDA_CHECK(cudaMemcpy(&ds, g_stats, sizeof(ds), cudaMemcpyDeviceToHost));
  g_last_stats.avg_density = ds.density_sum / kParticleCount;
  g_last_stats.max_density = bits_to_float(ds.density_max_bits);
  g_last_stats.avg_speed   = ds.speed_sum / kParticleCount;
  return g_last_stats;
}

void step_simulation(float dt, const MouseState &mouse, float4 *render_particles) {
  if (!g_initialized) init_simulation();

  int blocks    = (kParticleCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
  int num_cells = g_params.grid_w * g_params.grid_h * g_params.grid_d;

  apply_external_forces_kernel<<<blocks, kThreadsPerBlock>>>(g_vel, kParticleCount, dt);
  apply_mouse_force_kernel<<<blocks, kThreadsPerBlock>>>(g_pos, g_vel, kParticleCount, dt, mouse);
  predict_positions_kernel<<<blocks, kThreadsPerBlock>>>(g_pos, g_vel, g_predicted, kParticleCount, dt);
  enforce_box_boundary_kernel<<<blocks, kThreadsPerBlock>>>(g_predicted, g_vel, kParticleCount, 0);

  for (int iter = 0; iter < g_params.solver_iterations; ++iter) {
    rebuild_spatial_grid(g_predicted, blocks, num_cells);
    find_neighbors_grid_kernel<<<blocks, kThreadsPerBlock>>>(
        g_predicted, g_particle_index, g_cell_start, g_cell_end,
        g_neighbors, g_neighbor_counts, kParticleCount);
    compute_lambda_kernel<<<blocks, kThreadsPerBlock>>>(
        g_predicted, g_neighbors, g_neighbor_counts, g_lambda, g_density, kParticleCount);
    compute_delta_p_kernel<<<blocks, kThreadsPerBlock>>>(
        g_predicted, g_neighbors, g_neighbor_counts, g_lambda, g_delta_p, kParticleCount);
    apply_delta_p_kernel<<<blocks, kThreadsPerBlock>>>(g_predicted, g_delta_p, kParticleCount);
    enforce_box_boundary_kernel<<<blocks, kThreadsPerBlock>>>(g_predicted, g_vel, kParticleCount, 0);
  }

  update_velocity_and_position_kernel<<<blocks, kThreadsPerBlock>>>(
      g_pos, g_vel, g_predicted, kParticleCount, dt);
  enforce_box_boundary_kernel<<<blocks, kThreadsPerBlock>>>(g_pos, g_vel, kParticleCount, 1);

  run_post_step_passes(blocks, dt);
  export_render_particles_kernel<<<blocks, kThreadsPerBlock>>>(
      g_pos, g_density, render_particles, kParticleCount);
  CUDA_CHECK(cudaGetLastError());
}
