#include "kernel.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void update_points_kernel(float2* positions, int n, float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u = (float)i / (float)n;
    float angle = u * 2.0f * CUDART_PI_F * 8.0f + t;
    float radius = 0.25f + 0.55f * u + 0.04f * sinf(t * 2.0f + u * 30.0f);
    positions[i] = make_float2(cosf(angle) * radius, sinf(angle) * radius);
}

void launch_update_points(float2* positions, int n, float t) {
    int block = 256;
    int grid = (n + block - 1) / block;
    update_points_kernel<<<grid, block>>>(positions, n, t);
}
