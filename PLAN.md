# Water Simulation — Development Plan

**Method:** Position Based Fluids (PBF) — Macklin & Müller 2013
**Platform:** CUDA + OpenGL interop, native C++
**Trajectory:** 2D contained → 2D arbitrary scenes → 3D

## Tech stack

- **C++17** host code
- **CUDA** (kernels for sim)
- **OpenGL 4.5** for rendering, with CUDA↔GL buffer interop
- **GLFW** — window, keyboard, mouse
- **GLAD** — GL loader
- **GLM** — vector/matrix math (host side)
- **Dear ImGui** — parameter tuning UI
- **Thrust** — bundled with CUDA, used for sort/scan
- **CMake** as build system

## Phase 0 — Project scaffolding
- CMake project, link CUDA + OpenGL + GLFW + GLAD + ImGui
- GLFW window, OpenGL context, clear-color frame loop
- "Hello triangle" to confirm GL pipeline
- Allocate a CUDA buffer, register it with GL via `cudaGraphicsGLRegisterBuffer`, write from a kernel, read from a vertex shader — confirms interop works
- ImGui overlay with FPS counter

**Done when:** a CUDA kernel writes positions that an OpenGL shader renders as points.

## Phase 1 — 2D PBF, naive neighbors, box boundary
- Particle SoA buffers: `float2 pos`, `float2 vel`, `float2 predicted_pos`, `float lambda`, `float2 delta_p`
- Kernels:
  - `apply_external_forces` (gravity → velocity)
  - `predict_positions` (vel → predicted_pos)
  - `find_neighbors_naive` (O(n²) for now, fixed-cap neighbor list per particle)
  - `compute_lambda` (density constraint, eq. 8–11 in PBF paper)
  - `compute_delta_p` (eq. 12, with tensile instability term)
  - `apply_delta_p` (predicted_pos += delta_p)
  - `update_velocity_and_position` (vel = (predicted - pos) / dt; pos = predicted)
  - `enforce_box_boundary` (clamp + reflect velocity)
- Solver loop: 3–5 Jacobi iterations of (lambda → delta_p → apply)
- Render particles as point sprites with size based on density
- ~2k particles target

**Done when:** water settles in a box, drops in a column collapse and slosh.

## Phase 2 — Mouse interaction
- GLFW cursor + button callbacks → host-side mouse state struct
- Convert screen → world coords via inverse projection
- Pass `{mouse_pos, mouse_vel, mouse_radius, button_state}` as kernel uniform
- New kernel `apply_mouse_force`: for each particle within radius, add impulse along mouse velocity (drag) or radial outward (push)
- ImGui sliders for radius and force strength

**Done when:** clicking and dragging visibly pushes the fluid around.

## Phase 3 — Spatial hash for neighbor search
- Uniform grid, cell size = kernel radius (so the influence neighborhood of any particle fits inside the 3×3 block of cells around it)
- Per-particle cell hash kernel
- `thrust::sort_by_key` to sort particle indices by cell hash (positions/velocities stay in place; the sort produces a particle-index map per cell)
- Cell-start / cell-end index arrays via boundary detection kernel
- Replace naive neighbor search with 9-cell (2D) lookup
- Keep an A/B toggle (naive vs grid) in the ImGui panel during phase 3 so densities can be visually cross-checked, then drop the naive code once parity is established
- Target: 50k–100k particles at 60 FPS

**Done when:** particle count is 25× higher with no frame-rate hit.

## Phase 4 — 2D polish
- Vorticity confinement (eq. 15–16)
- XSPH viscosity (eq. 17)
- ImGui panel for parameter tuning, exposing: rest density, kernel radius, solver iterations, viscosity, vorticity epsilon, `lambda_epsilon` (CFM relaxation), and the tensile-instability terms (`tensile_k`, `tensile_n`, `tensile_q`)
  - Use text-box inputs (`ImGui::InputFloat` / `InputInt`) rather than sliders, so exact values can be typed and parameters that span several orders of magnitude (e.g. `tensile_k ≈ 3e-4`, `lambda_epsilon ≈ 400`) are easy to dial in
- Parameter profiles: save/load named presets to/from JSON on disk (e.g., "calm pool", "splashy", "viscous"); ImGui dropdown to switch profiles, "Save as…" and "Delete" buttons; profiles stored in `profiles/` next to the binary
- Optional: screen-space fluid surface (depth pass + bilateral blur + normal reconstruction)

**Done when:** the sim looks like water, and you can switch between saved looks without recompiling.

## Phase 5 — Arbitrary 2D scenes
- Signed distance field representation for static obstacles
- SDF baked from analytic primitives (circles, boxes, polygons) into a 2D texture
- Sample SDF in the boundary kernel: if `sdf < 0`, push particle out along `∇sdf`
- Scene-loading from JSON
- Optional: dynamic rigid bodies with two-way coupling (Macklin "Unified Particle Physics" approach)

**Done when:** you can drop water into any user-defined 2D scene.

## Phase 6 — 3D port
- `float2` → `float3` everywhere (vectors, kernels, hash)
- Spatial hash: 27-cell lookup
- 3D orbit camera + GLFW mouse-drag camera control
- Render as 3D point sprites with depth, basic Phong on impostor spheres
- Reuse same boundary SDF approach (now 3D textures)
- Target: 200k–500k particles real-time

**Done when:** 3D water in a 3D box, mouse rotates camera, can still push fluid via raycast from cursor.

## Phase 7 — 3D rendering polish
- Screen-space fluid rendering (Müller "Screen Space Fluid Rendering with Curvature Flow")
  - Particle depth pass
  - Bilateral / curvature-flow smoothing on depth
  - Normal reconstruction from depth derivatives
  - Thickness pass (additive blending)
  - Final shading: refraction, Fresnel, absorption (Beer-Lambert)
- Skybox / environment for reflections
- Nsight Compute profiling pass; tune block sizes, occupancy

**Done when:** it looks like water in a render, not like dots.

## Key references
- Macklin & Müller, *Position Based Fluids*, SIGGRAPH 2013
- Macklin et al., *Unified Particle Physics for Real-Time Applications*, SIGGRAPH 2014 (for Phase 5+ rigid coupling)
- Müller et al., *Screen Space Fluid Rendering with Curvature Flow*, I3D 2009 (Phase 7)
- Hoetzlein, *Fast Fixed-Radius Nearest Neighbors* (spatial hash on GPU, Phase 3)
- Green, *Particle Simulation using CUDA* (NVIDIA whitepaper, Phase 3 reference impl)

## Out of scope (for now)
- Multi-phase fluids (oil + water)
- Surface tension beyond Macklin's tensile term
- Compressible / smoke / fire
- Multi-GPU
- Mesh export / offline render
