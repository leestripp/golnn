# GameOfLifeNN Improvement Plan

This roadmap builds on the completed simulation-core refactor. The priorities are:
more agents at a smooth frame rate, a real persistent CPU thread pool, standardized
brains that can be processed efficiently, richer senses and learning, a more stable
ecosystem, and eventually a Vulkan renderer.

The CPU remains the simulation platform. Vulkan is for rendering first; GPU neural
network inference/training is a separate, optional experiment only after the CPU work
has established clean batching, measurements, and a genuine need for it.

## Current Status (2026-10-26)

The original core refactor provided C++23, presets, tests, flat `float` matrices,
reused scratch buffers, memoized forward passes, spatial grids, and `golnn_bench`.

This implementation round completed the first tranche of the roadmap:

- `SimulationConfig` now controls seed, caps, predator ratio, worker count, regrowth,
  and optional CSV telemetry. All global/`rand()` RNG was removed in favor of seeded
  simulation and per-agent streams.
- Full-world checksums cover transforms, food, brain weights/state, and gradients.
  Seeded runs and serial/parallel runs replay identically.
- `lsRNN` history is a reusable logical ring; MLP backward and interaction scratch
  no longer allocate steadily; the second food-grid rebuild was eliminated.
- **The persistent thread pool is implemented and enabled by default.** It uses
  start-of-tick transform snapshots, dynamically sized index chunks, main-thread
  participation, serial interactions/reproduction, RAII shutdown, and TSan tests.
- Adaptive vegetation regrowth is the default, with a bounded population/consumption
  controller and emergency floor. The original fixed mode remains available.
- Optional CSV telemetry and a live GUI ecology counter row are implemented.
- Nine CTest tests pass in Release, ASan, and UBSan; the threading suite is TSan-clean.

Reference scaling (`300 1000 500 100 --seed 42`, before the ecology milestone):

| workers | ms/tick | ticks/s | speedup |
|---|---:|---:|---:|
| 1 | 18.78 | 53 | 1.0× |
| 2 | 10.56 | 95 | 1.8× |
| 4 | 6.67 | 150 | 2.8× |
| 8 | 5.68 | 176 | 3.3× |
| 12 | **5.63** | **178** | **3.3×** |

The plateau above four workers was expected from heterogeneous brain sizes and the
serial interaction/grid phases.

### Standardized brain tiers — completed

`BrainCatalog.h` now defines small `{32,32,32}`, medium `{48,48,48,48}`, and large
`{64,64,64,64,64}` movement networks, with 12/24/32-neuron state brains. Tiers are
inherited, mixed populations are batched independently, and custom brains retain a
scalar reference path. Layered batch forward/backward passes are exactly checksum-
equivalent to scalar execution. GUI keys **1/2/3** select the tier for new agents;
existing agents retain their tier. CSV telemetry includes per-tier population counts.

Current 1,000-agent results with 12 workers (`300` ticks, seed 42):

| brain tier | ms/tick | ticks/s |
|---|---:|---:|
| small | **4.07** | **245** |
| medium | 7.85 | 127 |
| large | 14.07 | 71 |

The small tier improves the previous 5.69 ms/tick result by another ~1.4×. Batching
itself is approximately performance-neutral so far (4.07 vs 4.10 ms scalar); the
near-term gains come from standardized shapes. SoA/SIMD remains optional future work.
All tiers completed healthy 3,000-tick ecology runs. Release, ASan, UBSan, and TSan
suites pass, including exact scalar/batched replay tests.

### Native Vulkan renderer — completed

Vulkan is now the default backend on native Wayland, with the existing instanced OpenGL
renderer retained as a fallback and through `GOLNN_RENDERER=opengl`. GTK4 intentionally
exposes no Vulkan internals, and its top-level surface already owns DRM-sync state, so
presenting directly onto it fails. `WaylandSubsurface` instead creates an independent
child surface for the scene area; GTK retains the controls/statistics and input.

The backend includes depth-tested render passes, world-cube edges, food points,
interpolated instanced agents, per-frame host-visible snapshot/uniform buffers, and
swapchain recreation. GLSL sources and validated generated SPIR-V arrays are checked in;
`tools/compile_vulkan_shaders.sh` regenerates them after edits without making shader
tools a normal build dependency.

Khronos validation reports no messages at 100 or 1,000 agents, including resize and
normal GTK shutdown. All ten tests pass. A coarse 1,000-agent process comparison used
about 110% total CPU for Vulkan versus 156% for OpenGL (simulation included); this is
not a controlled FPS result. A stochastic RNN test was seeded, and OpenGL unrealize was
fixed to use its live context directly. Remaining cleanup: a common abstract renderer
interface and explicit FPS/GPU-time instrumentation if deeper comparison is needed.

### Richer sensor system — completed

Perception layout v2 (`Perception.h`) preserves channels 0-30 and appends normalized
velocity, heading, teacher-goal direction/alignment, smoothed energy trend, local food
density, continuous boundary proximity, and threat proximity. The state brain retains
its original 12 bins plus five compact context inputs. Vision covers 300 degrees with
close peripheral awareness; scent and hearing remain omnidirectional.

Food vision/density/scent now share one grid visit, and predator prey vision/scent also
share a visit. Goal channels are derived from the exact teacher target, including
boundary avoidance. Interaction sounds now use a previous-tick audible buffer, fixing
the old lifecycle where they were cleared before perception. Agent cleanup compacts
in place and reuses offspring scratch rather than replacing the whole allocation.

All three tiers completed healthy 3,000-tick ecology runs. The richer 45-channel input
runs at roughly 4-4.3 ms/tick for 1,000 small-tier agents with 12 workers. New tests
cover layout stability, goal alignment, velocity, rear-blind-spot vision versus scent,
boundary proximity, and one-tick sound lifetime. Full sanitizer and GUI checks pass.

### Decoupled instanced rendering — completed

The simulation now runs on a dedicated `std::jthread` with bounded catch-up and clean
GTK shutdown. Immutable `WorldSnapshot` objects carry float instance transforms,
previous transforms, colors, ecology statistics, and UI settings. Two reusable slots
avoid steady-state allocation; a held slot is never mutated. Headless benchmarks
disable publication and remain at approximately **4.06 ms/tick** for 1,000 small-tier
agents with 12 workers.

The OpenGL backend now draws all agents in **one instanced draw call**, uploads food
from direct float buffers, keeps static grid geometry, and interpolates current and
previous transforms in the vertex shader. GTK no longer expands agents into nested
`double` vertices or holds the world mutex. A concurrent-reader/immutability test was
added; all ten tests pass in Release, ASan, UBSan, and TSan, and the instanced GUI
smoke test succeeds on Wayland.


## Rules for Every Milestone

- Keep each milestone independently buildable, testable, and revertible.
- No behavior changes in performance-only milestones. Ecology, sensors, learning, and
  brain-topology changes are explicitly behavioral and evaluated with telemetry.
- Build warning-clean; run all CTest tests. Run ASan/UBSan for structural changes and
  TSan for threading changes.
- Benchmark release builds with several runs and report the median, not the best run.
  Track ms/tick, ticks/s, final populations, and memory use.
- Add deterministic seeded scenarios before parallel or batched execution. A fixed seed
  must reproduce a single-threaded run exactly unless a documented behavior change is
  being tested.
- Prefer standard-library facilities and existing dependencies. Vulkan SDK/loader
  integration is approved as part of the renderer milestone; any additional library
  still needs explicit approval.
- Preserve the dual-brain design and the emergent character of the simulation. The aim
  is not to replace interesting behavior with a scripted ecosystem.

---

## M0 — Reproducibility, Correctness, and Performance Oracle

Make measurements trustworthy before introducing threads or changing brain layouts.

- [x] Replace file-static/global RNG use, including `rand()` in `ls3DRNN`, with explicit
      seeded engines. Use per-agent RNG streams for work that will later run in parallel.
- [ ] Add a `SimulationConfig` for seed, world size, population/food caps, spawn ratio,
      tick limits, and worker count. Keep current values as defaults.
- [ ] Extend `golnn_bench` with repeatable scenarios, warmup, multiple samples, median
      and spread, peak RSS where practical, and optional telemetry output.
- [x] Add fixed-seed regression scenarios at small and saturated populations. Include
      enough ticks to exercise births, deaths, meat creation, predation, and regrowth.
- [x] Reserve stable agent storage and remove raw-pointer lifetime assumptions across
      phases. Document the storage/invalidation policy.
- [ ] Add numerical-gradient tests for both neural cores, grid-vs-brute-force sensor
      tests, reproduction/mutation bounds, and deterministic replay checks.

**Acceptance:** identical seeded single-threaded runs produce identical checksums and
statistics; benchmarks are repeatable; ASan/UBSan and all tests pass.

## M1 — Finish Low-Risk CPU Hot-Path Work

Squeeze out avoidable work before changing execution topology.

- [x] Remove remaining per-step allocations/copies in MLP backward propagation,
      forward-return paths, delta propagation, history eviction, and food compaction.
      Reuse capacity; use spans/views at internal API boundaries.
- [x] Replace the bounded-but-growing history behavior with a fixed ring/window sized
      to actual BPTT needs.
- [x] Rebuild only what changes between perception and interactions: agent grids need
      refreshing after movement; static food grids do not.
- [ ] Profile allocation counts, cache behavior, `tanh`, quaternion/matrix conversion,
      and grid queries. Optimize measured hotspots rather than adding speculative SIMD.
- [ ] Add a strict headless stress scenario above the normal population cap during
      development to expose quadratic behavior and memory growth.

**Acceptance:** no steady-state per-tick heap growth, no behavior delta in seeded tests,
and a measured improvement or a documented reason to keep each optimization.

## M2 — Persistent CPU Thread Pool (High Priority)

Parallelize agent think/move work without making world interactions racy.

- [x] Implement a small persistent thread pool owned by `Simulation` or the app-level
      simulation runner. Create workers once, not per tick; use condition variables,
      RAII shutdown, and a main-thread participation option.
- [ ] Split a tick into explicit phases:
      1. build/read the world snapshot and spatial grids;
      2. in parallel, update independent agents (perceive, train, move);
      3. wait for all workers;
      4. rebuild moved-agent grids;
      5. resolve eating/predation serially in a stable order;
      6. process deaths/reproduction serially;
      7. publish an immutable render/telemetry snapshot.
- [x] Never let workers mutate `Simulation`, food, sound events, other agents, shared
      RNGs, the tokenizer, or shared scratch memory. Queue spawn commands for the main
      thread rather than locking the whole world around workers.
- [x] Give each agent its own RNG/state and scratch buffers. Avoid shared static mutable
      initialization in agent updates.
- [x] Use stable index ranges and grain sizes large enough to amortize synchronization.
      Benchmark worker counts `1, 2, 4, ... hardware_concurrency`; support an environment
      or CLI override and retain a single-thread fallback.
- [x] Do not use `std::execution::par` as a substitute for a persistent pool and do not
      add a mutex to every neural operation.
- [ ] Add a TSan preset and stress tests covering shutdown, spawn-during-run, population
      compaction, reproduction, and zero/one-agent cases.

**Acceptance:** TSan-clean; single-worker results match serial results; multi-worker
runs preserve phase ordering and (with per-agent streams) seeded results; near-linear
scaling initially, with a documented plateau. Target at least **2× simulation throughput
at 1000 agents** on the available multicore CPU, without GTK frame stalls.

## M3 — Standardized Brain Tiers and Batched CPU Math

Random per-agent layer counts/sizes defeat efficient batching. Retain diversity through
several controlled architectures rather than arbitrary geometry for every agent.

- [x] Define a small versioned catalog, initially for example:
      `small` (current average cost), `medium` (moderately larger), and `large`
      (experimental). Fix exact input/output contracts and layer dimensions after
      measuring current distributions; do not treat these names as final dimensions.
- [x] Every newborn selects/inherits a tier. Mutation changes weights and permitted
      hyperparameters, not tensor shapes. Reproduction preserves a valid tier.
- [x] Group agents by brain tier and process each group in batches. Keep the existing
      single-agent implementation as the correctness reference.
- [ ] Move weights, biases, activations, and gradient accumulators toward contiguous
      tier-owned storage/SoA layout. Pack per-agent inputs and outputs into contiguous
      batch buffers; avoid handles that dangle during population compaction.
- [x] Implement batched forward and backward kernels over standardized shapes. Start
      with compiler auto-vectorization; add AVX2/runtime dispatch only if profiling
      justifies it, with a scalar fallback.
- [x] Keep the movement network and state network separate, but use the same matrix
      kernel machinery where dimensions allow.
- [x] Compare quality as well as speed: survival, births, state diversity, and learning
      progress must not collapse merely because architectures are standardized.
- [x] Enable moderately wider/deeper `medium`/`large` tiers after standardized
      shapes prove viable. Bigger brains are an experiment, not an automatic intelligence improvement.

**Acceptance:** scalar-reference numerical equivalence within float tolerance; no cache
invalidation bugs after optimizer steps/mutation; clear throughput gains per tier; larger
brains can be enabled without changing call sites or breaking saved scenarios.

## M4 — Telemetry and Adaptive Food Regrowth

First observe the ecosystem, then add bounded feedback so feeders do not repeatedly
crash from an inflexible fixed trickle of vegetation.

- [ ] Emit optional CSV or JSONL telemetry every configurable N ticks: tick, seed,
      feeder/predator counts, births/deaths and causes, food by type, food eaten,
      mean/min/max energy and age, generation, state histogram, teacher-forced share,
      brain-tier counts, and sim/render timing.
- [x] Add a lightweight GUI statistics panel and/or live counters driven from the same
      snapshot data; do not make workers perform I/O.
- [x] Measure vegetation availability per feeder and consumption/regrowth rates over a
      sliding window. Include meat separately rather than treating all food as equal.
- [x] Replace or augment fixed `5 food / 300 ticks` regrowth with a bounded controller
      based on feeder population and a target vegetation-per-feeder band. Account for
      recent consumption and use smoothing/hysteresis to prevent oscillation.
- [x] Enforce `MAX_FOOD`, maximum spawn-per-tick, world bounds, and a minimum emergency
      regrowth floor when feeders remain but vegetation is exhausted.
- [x] Keep an ecology configuration preset for the original fixed regrowth behavior so
      old and new experiments remain comparable.
- [x] Monitor predator/prey ratio and extinction risk in telemetry. Do not add automatic
      predator rescue until food-feedback experiments show it is necessary.

**Acceptance:** long seeded runs show fewer avoidable vegetation-collapse extinctions,
no runaway food/population oscillation, bounded caps, and reproducible telemetry. Compare
fixed and adaptive regimes over identical seed sets.

## M5 — Better Senses and Decision Inputs

Add useful information, not merely more input dimensions.

- [x] Preserve direction-vector teaching; never accidentally teach absolute positions.
- [x] Version the perception layout and document every channel, range, coordinate space,
      and missing-target value. Add layout tests for feeder and predator perceptions.
- [x] Add normalized velocity/heading, target-relative heading, energy trend over a short
      window, local food density, and clearer boundary proximity. Reuse grid queries and
      scratch buffers.
- [x] Distinguish fields of view: directional vision versus wider scent/hearing. Consider
      nearest-K or multi-ray target summaries so one closest object does not hide the
      local crowd/food distribution.
- [x] Improve sound events with finite lifetime/intensity falloff if telemetry shows the
      current one-tick signal is rarely useful.
- [x] Feed a compact, normalized subset to the state brain; avoid overwhelming its small
      hidden state with redundant channels.
- [x] Evaluate sensory changes by behavior metrics and observation: fleeing earlier,
      less wall sticking, better food acquisition, and non-degenerate state usage.

**Acceptance:** perception unit tests and normalization checks pass; added sensors produce
measurable behavioral value that justifies their compute cost.

## M6 — Learning, Evolution, and Persistence

- [ ] Anneal teacher forcing gradually per agent/generation, with safe fallback and
      telemetry for the teacher-to-autonomy transition.
- [ ] Add temperature-controlled state exploration instead of permanent hard argmax;
      anneal exploration and log state occupancy.
- [ ] Detect hidden activation saturation and stabilize training before increasing brain
      size. Add gradient/weight clamps or normalization only when measurements support it.
- [ ] Experiment with crossover and evolvable bounded hypergenes (learning rate, batch
      size, metabolism), one change at a time. Preserve clone+mutate as a baseline.
- [ ] Save/load complete scenarios: config, seed/RNG state, agent tier, weights, brain
      state, transform, energy, age, and generation. Use explicit format versions.
- [ ] Add checkpoints for interesting evolved populations so behavior can be replayed and
      compared across code changes.

**Acceptance:** agents increasingly survive/act without teacher assistance across tested
seed sets; no universal state collapse; saved scenarios round-trip and replay correctly.

## M7 — Decouple Simulation and Rendering; Optimize OpenGL First

This is both an immediate GUI performance improvement and preparation for Vulkan.

- [x] Run the fixed-timestep simulation independently of GTK rendering, with bounded
      catch-up and no uncontrolled spiral after a stall.
- [x] Publish double-buffered immutable render snapshots; remove the renderer's need to
      hold `m_data_mutex` throughout simulation work.
- [x] Replace nested `vector<double>` render data with compact float instance records
      (transform, color, type). Avoid rebuilding every tetrahedron vertex on the CPU.
- [x] Interpolate `m_prev_transform` to the current transform for smooth rendering at a
      different rate from simulation ticks.
- [x] Use instanced agent rendering and persistent/static grid geometry in the existing
      OpenGL backend. Keep food in an efficient point/instanced path.
- [ ] Introduce a narrow renderer interface consuming the same render snapshot, with an
      OpenGL implementation retained as fallback/reference.

**Acceptance:** simulation throughput is independent of render FPS; no lock convoy or
tearing; instanced OpenGL reduces render CPU time and provides a correctness reference
for Vulkan.

## M8 — Vulkan Renderer

Implement Vulkan only after M7 isolates rendering from simulation.

- [x] **Technical spike first:** choose the supported GTK4/native-window presentation
      path on this Wayland machine. `Gtk::GLArea` is an OpenGL widget, so do not assume a
      Vulkan swapchain can simply be dropped into it. Prototype surface creation,
      resizing, and shutdown before migrating scene code.
- [x] Add CMake detection and an opt-in Vulkan backend. Keep OpenGL builds working when
      the Vulkan SDK/driver is unavailable.
- [x] Implement instance/device/surface selection, queues, swapchain, depth buffer,
      command pools/buffers, synchronization, and resize/out-of-date recovery with RAII.
- [x] Port the grid, instanced agents, and food using SPIR-V shaders and the shared float
      instance snapshots. Use dynamic vertex/uniform/storage buffers as appropriate;
      avoid per-frame allocation and unnecessary readbacks.
- [x] Handle frames in flight correctly and ensure simulation snapshot publication never
      waits on GPU readback.
- [ ] Enable Vulkan validation layers in debug builds; test minimize, resize, unrealize,
      backend fallback, and clean shutdown on Wayland.
- [ ] Add backend/FPS/simulation-rate indicators and compare CPU render time, GPU frame
      time, latency, and memory against instanced OpenGL.

**Acceptance:** feature parity with the OpenGL renderer, validation-clean frames, robust
resize/shutdown, stable simulation timing, and a measured rendering benefit at high agent
counts. If GTK presentation proves unsuitable, stop at the spike and choose a minimal
windowing architecture change explicitly rather than accumulating hacks.

## M9 — Optional GPU Compute Investigation (Not Scheduled)

Revisit only if standardized batched CPU brains still limit the desired population.

- [ ] Start with a standalone Vulkan-compute benchmark for one standardized tier and one
      matrix operation; compare against the optimized CPU kernel at realistic batch sizes.
- [ ] Include upload/download, synchronization, and training-gradient costs in the
      measurement. Inference-only speedups are not enough if per-tick training dominates.
- [ ] Prefer persistent resident weights/buffers and whole-tick batching if an experiment
      proceeds. Do not dispatch one tiny draw/dispatch per agent.
- [ ] Keep the CPU backend and deterministic reference tests.

GPU compute is a **go/no-go experiment**, not a promised feature. Small heterogeneous
networks and frequent host interaction can make it slower than the optimized CPU path.

---

## Suggested Delivery Order

1. M0 reproducibility and tests.
2. M1 low-risk CPU cleanup.
3. **M2 persistent thread pool.**
4. M3 standardized/batched brain tiers.
5. M4 telemetry and adaptive vegetation.
6. M7 render decoupling and instanced OpenGL (can overlap M4/M5 after the snapshot API).
7. M5 senses and M6 learning/evolution as measured behavior experiments.
8. M8 Vulkan renderer.
9. Optional M9 GPU-compute spike only with new profiling evidence.

This order gets more agents and a healthier observable ecosystem early, while creating
the exact data layouts and renderer boundary needed for larger brains and Vulkan later.
