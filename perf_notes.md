  High-impact

  1. Reorder particles into sorted order (biggest win). Right now compute_cell_hash_kernel sorts
  particle_index by cell_hash, but positions[] itself stays in original order.The inner loop at
  kernel.cu:285-293 does positions[particle_index[idx]] — a gather that defeats coalescing and the
  cache. After the sort, copy positions[] (and any other per-particle dataaccessed during traversal)
   into a sorted buffer, then have neighbor traversal read positions_sorted[idx] directly. The
  downstream particles in the same cell live at consecutive indices → coalesced loads, much higher
  L1/L2 hit rate. Same change benefits all five other kernels that traverse the grid (density,
  lambda, delta, vorticity, viscosity).

  2. Skip the neighbor list entirely (NVIDIA Flex style). You're spending ~`N * kMaxNeighbors * 4`
  bytes of writes (96 ints × N) just to be re-read by density/lambda/delta. Modern PBF
  implementations re-traverse the grid in each of those kernels instead. With (1)above, traversal is
   cheap, and you delete a kernel + a huge buffer + the neighbor-cap truncation issue. Worth
  profiling against keeping it; if traversal is bandwidth-bound, the list paysfor itself, otherwise
  drop it.

  3. Precompute clamped loop bounds. Replace the per-cell if (nz<0||nz>=grid_d) continue; checks
  with:
  int z0 = max(c.z-1, 0), z1 = min(c.z+1, grid_d-1);
  and similarly for y/x. Removes ~27 branches per particle and lets the compiler unroll cleanly.

  Medium-impact

  4. const __restrict__ + __ldg. Mark positions, cell_start, cell_end, particle_in  delta, vorticity, viscosity).

  5. Morton/Z-order cell hashing. Replace nz*W*H + ny*W + nx with a Mortoncode.
  Particles that are spatial neighbors get nearby indices, so when processed in block
  order they hit warmer caches. Cheap change, measurable on large grids.

  6. Shared-memory cell-start tiling. When a thread block's particles share most of
  the same 27 cells (likely after sorting), have the block cooperatively load each
  cell's [start,end) once into shared memory rather than every thread re-fetching from
   global.

  Smaller / situational

  7. Drop kMaxNeighbors if oversized. If actual counts top out around 40–50, dropping
  the cap from 96 reduces write bandwidth proportionally — quick win if you keep the
  list.

  8. Replace j == i branch with index comparison after sort. With (1), i becomes the
  sorted index — the self check is just idx == i, no indirection.

  9. Profile before deciding. Run with nsys/ncu to confirm whether you're
  memory-bandwidth bound (then 1, 2, 5 dominate) or instruction-bound (then 3, 8
  matter more).

  My recommendation: start with (1) — it's a contained change, gives most other
  kernels a free speedup, and is a prerequisite for (2). Want me to draft the patch?
