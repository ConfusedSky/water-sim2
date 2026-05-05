#!/usr/bin/env bash
# Forces OpenGL onto the NVIDIA GPU via PRIME render offload so CUDA-GL
# interop can share buffers. Required on hybrid-graphics laptops where
# the X server otherwise picks the integrated GPU.
set -e
cd "$(dirname "$0")"
exec env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
    ./build/water_sim2 "$@"
