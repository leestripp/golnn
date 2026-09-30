# Plan.md — C++23 Refactor: Performance & Agent Quality

Refactor roadmap for GameOfLifeNN (`golnn`). Goal: modernize to C++23, make the
simulation run faster (bigger populations at 30+ fps), and produce smarter / more
stable evolving agents — without changing the emergent-behavior character of the sim.

Everything here is a proposal. Milestones are ordered so the tree stays buildable
and behavior-preserving (or intentionally changed) after each one.

---

## 0. Baseline & Tooling First — ✅ DONE

Before touching behavior, establish measurement and safety nets.

- [ ] Bump `CMAKE_CXX_STANDARD` 20 → **23** (gcc 12+/clang 16+ already handle the subset we need).
- [ ] Add `CMakePresets.json` (`debug`, `release`, `asan`, `ubsan`) so every dev/agent
      configures the same way.
- [ ] Warning hygiene: add `-Wpedantic -Wconversion -Wshadow -Wdeprecated` to a
      `golnn-warnings` **interface target** and ratchet down; aim for `-Werror` in CI later.
- [ ] Sanitizer builds (`-fsanitize=address,undefined`) wired into presets; run the sim
      headless-style for N ticks under ASan before each milestone merges.
- [ ] Add a **headless benchmark mode**: `golnn --bench <ticks>` runs `Simulation::tick()`
      without GTK/GL (extract sim core from `main.cpp` so it links without a display).
      Record ticks/sec, agents, food, per-tick ms — this is our regression oracle.
- [ ] Tiny smoke test executable (CTest) that constructs `Simulation` and ticks it 100×
      — no framework needed yet; prove the loop is safe under ASan/UBSan.

## 1. Hot-Path Performance — Neural Cores (biggest win) — ✅ DONE

The RNNs are the per-tick, per-agent cost. Current design: `vector<vector<double>>`
weights, fresh heap allocations in `forward_pass`, `double` everywhere.

- [ ] **Flat contiguous weight storage.** Replace `vector<vector<double>>` with a single
      `std::vector<float>` per matrix + `std::mdspan`/custom 2D view (row-major). One
      allocation per matrix, cache-friendly inner loops, auto-vectorizer friendly.
- [ ] **`float`, not `double`.** Simulation tolerances don't need double precision;
      halves memory traffic and doubles SIMD width. Keep `double` only in
      loss/reporting paths if drift shows up.
- [ ] **Zero-allocation forward pass.** Pre-allocate `h_new`, `output`, scratch buffers
      in the constructor; `forward_pass` writes into member scratch (`std::span` views).
      Remove per-call `vector` churn.
- [ ] **Unbounded history bug.** `lsRNN::m_h_history` grows every forward pass and is
      only cleared by `reset_state()` — for long-lived agents this is a memory leak
      that also slows BPTT. Bound history to the BPTT window (or a ring buffer of
      `BATCH_SIZE + 1`), fix `reset_state` to be called by `Agent` on a schedule.
- [ ] **Bug: `find_min_max` uses `numeric_limits<double>::min()`** (smallest positive
      normal, not lowest) — normalizing an all-negative vector produces garbage. Use
      `lowest()`. Add a unit test.
- [ ] **Fuse loops in `forward_pass`.** Compute `h_new = b_h + W_xh·x + W_hh·h` as two
      strided axpy passes; same for `W_ho·h`. Consider loop tiling for 64-neuron layers.
- [ ] **Batch math, not batch API.** `Agent::BATCH_SIZE = 30` already accumulates
      gradients per tick — keep that, but make the accumulation loops flat/soa so the
      optimizer step is one linear sweep over `float`s.
- [ ] Optional/later: evaluate a hand-rolled SSE/AVX2 kernel or `std::experimental::simd`
      behind a CMake option (`GOLNN_ENABLE_SIMD`), gated by benchmark results.

**Acceptance:** bench ticks/sec improves ≥ 2× at 1000 agents; ASan/UBSan clean; behavior
delta within run-to-run noise (same seed harness).

## 2. Simulation Data Structures — Kill the O(n²) — ✅ DONE

Perception and interactions scan all agents × all food every tick. At 1000 agents /
1000 food that's ~10⁶+ distance checks per tick at 30 fps.

- [ ] **Uniform spatial hash grid** (cell size ≈ max sense radius) over the world,
      rebuilt each tick, storing agent/food indices. Neighbor queries become O(k).
- [ ] Rewrite `Agent::perceive` (food direction, threat direction, friend density)
      and `Simulation::handle_agent_interactions` to use grid queries.
- [ ] Split `m_food` into **separate `vegetables` / `meat` vectors** (or one vector +
      type-partitioned index list) — every eat loop currently skips the wrong type.
- [ ] Replace `m_food` rebuild-per-tick with `std::erase_if` in place.
- [ ] **SoA or handle-stable agent storage.** `std::vector<Agent>` invalidates
      `Agent*` on growth; interactions currently hold raw pointers across phases.
      Either reserve `MAX_AGENTS` up front, or store `std::deque<Agent>` /
      `unique_ptr<Agent>` for pointer stability — pick one and document it.
- [ ] Reserve `surviving_agents` correctly (currently `reserve(m_agents.size())` is
      fine, but the dead-agent branch drops feeders into `m_food` — combine passes to
      avoid double iteration).
- [ ] Multi-threading (optional, milestone 2.5): agent updates are independent per
      tick — parallelize `for (auto& a : m_agents) a.update(world)` with
      `std::execution::par` or a small thread pool, keeping interactions single-threaded.
      Gate behind a CMake option; **determinism loss must be documented** (reseeded runs
      diverge — that's acceptable for an alife sim, but say so).

**Acceptance:** per-tick cost scales ~linearly with agent count on the bench; bench
oracle unchanged vs milestone 1 (modulo RNG stream changes — document seed changes).

## Results (M0–M2 complete)

A/B against `git HEAD` (both `-O3`, `golnn_bench 300 1000 500 100`):

| build | ms/tick | ticks/sec |
|---|---|---|
| original | 32.8 | 30.4 |
| M1 flat-float NN | 28.3 | 35.4 |
| + M2 spatial grid | 25.7 | 38.9 |
| + forward memoization + partitioned grids + alloc removal | **18.1** | **55.2** |

Net: **~1.8× at 1000 agents**, behaviour within noise (survivor/food counts
match the original run). gprof after M2: remaining time is MLP backward (32%),
MLP forward (26%), perceive (24%) — i.e. genuine learning compute, not waste.
The original 2× estimate assumed the NN cores dominated; at 1000 agents the
O(n²) perception did (fixed in M2), and the NN cores were only ~⅓ of runtime.

### M1 notes
- `lsRNN`: flat contiguous float weights (one allocation per matrix), zero-alloc
  forward via reused scratch, history bounded at 1024 steps with old-half
  eviction, `find_min_max` min→lowest() bug fixed. Added `train_step()` for
  allocation-free length-1 BPTT.
- `ls3DRNN`: flat row-major per-layer float weights, reused activation cache,
  flat gradient accumulators. `forward()` memoized — Agent calls train_step()
  then process_step() with the same input every tick, so the second forward is
  skipped (cache invalidated on weight changes).
- Fixed `prepare_mlp_input` header/impl mismatch (now void(command, out)).

### M2 notes
- New `SpatialGrid` (uniform grid, cell = 30 units = scent radius; `visit()` is
  a template over a box query, no allocations after construction).
- `Simulation` holds four type-partitioned grids (feeders / predators /
  vegetables / meat), rebuilt twice per tick (before updates, after movement).
- `Agent::perceive` rewritten: each sensor visits only the grid holding the
  types it can detect; friendly-density rides along the follow-range visit,
  hostile-density along the threat visit.
- `train_brain` reuses `m_food_pos` instead of re-scanning food
  (`find_closest_food_direction` deleted).
- `perceive()` and `train_state_brain()` write into reused member scratch
  buffers — no per-tick heap allocations on the hot path.

## 3. C++23 Modernization & Code Quality

Mechanical, low-risk, high-readability payoffs across the codebase.

- [ ] Replace `cout`/`printf`-style logging with **`std::print` / `std::format`**
      (alpha-agent report, debug output).
- [ ] Use **`std::expected`** for fallible operations (brain loading/saving later,
      parameter parsing) instead of magic sentinels like `learning_rate = -1.0f`.
- [ ] `std::string_view` for string parameters; `std::span<const float>` for
      perception/command vectors crossing APIs (kills the `vector<float>` copies in
      `process_step` call sites).
- [ ] `std::ranges` for the filter/sort/map pipelines (food compaction, closest-target
      scans, alpha-agent stats).
- [ ] Replace magic numbers with `static constexpr`/named constants (many already
      exist — extend to `main.cpp` camera speeds, food spawn scales).
- [ ] Remove `using namespace std;` from **headers** (`Agent.h`, `lsRNN.h`,
      `ls3DGraph.h`, `ls3DRNN.h`); qualify at call sites. Keep style consistent.
- [ ] Consistent include guards → `#pragma once` project-wide (one mechanical commit).
- [ ] Fix `Food::m_food_type` (`size_t` holding an enum) → proper `FoodType` member;
      `enum class FoodType` while at it.
- [ ] Replace `static` file-scope RNG engines with a passed-in `std::mt19937_64&` —
      testability + thread-safety story; single seeded engine option for reproducible runs.
- [ ] CMake modernization: `target_include_directories`/`target_link_libraries` on
      `golnn` (drop global `include_directories`), `file(GLOB CONFIGURE_DEPENDS)`.

## 4. Agent Quality — Learning & Evolution

The brains are the product. Current weaknesses: teacher forcing never anneals
(agents stay dependent on the teacher), mutation-only evolution (no crossover),
tanh-saturation risk, and state brain limited to binned energy/distances.

- [ ] **Anneal teacher forcing.** Add `teacher_probability` per agent: 1.0 at birth,
      decaying (e.g. ×0.999/tick, or by generation number). Late-life agents rely on
      their own RNN; evolution selects for brains that took over successfully.
      Log the crossover point as a metric.
- [ ] **Crossover on reproduce.** `Agent::reproduce` currently clones + mutates.
      Add weight-interchange crossover: pick per-layer (or per-weight with p=0.5)
      from either parent — requires keeping a second parent around; simplest version:
      pairwise nearest-energy mate, else clone+mutate as today.
- [ ] **Evolvable hypergenes.** Mutate (slightly, bounded) per-agent: `m_learning_rate`,
      hidden-layer count, neurons/layer, `BATCH_SIZE`, energy-decay — within the existing
      `MAX_LAYER_COUNT`/`MAX_NEURON_COUNT` caps. The Alpha-Agent report already prints
      these; make selection genuinely tune them.
- [ ] **Softer exploration in the state brain.** Instead of hard argmax over
      `IDLE/ROAM/HUNT/FLEE`, sample from the softmax (temperature annealed with age);
      prevents lock-in of a single state early in life.
- [ ] **Richer state-brain inputs.** Add: current velocity heading vs. target heading,
      energy delta over last N ticks, food density (not just agent density), boundary
      proximity scalar. Cheap with the spatial grid from milestone 2. Keep input dim
      small (<32) so the tiny RNN can actually learn it.
- [ ] **Stabilize `ls3DRNN` training.** The hidden state *is* the 4×4 transform —
      teacher-forced gradients can corrupt the rotation basis. After each
      `set_state`/train step, re-orthonormalize the rotation block (Gram–Schmidt or
      quat→matrix round trip); keeps matrices pure rotation + uniform-ish scale.
- [ ] **Anti-saturation.** Track `|h| → 1` in `lsRNN`; if hidden activations saturate,
      rescale incoming weights (poor-man's layer norm) or switch activation to
      `tanh` with leaky fallback. Measure: mean hidden activation magnitude over time.
- [ ] **Ecosystem balance levers.** Food regrowth currently fixed 5/300 ticks.
      Make it proportional to feeder population (prevents feeder extinction spirals)
      and expose predator/prey spawn ratio (currently hardcoded 10%) as a sim parameter
      with CLI/config override.
- [ ] **Observability.** Cheap CSV/JSONL telemetry every N ticks: population by type,
      mean energy, mean age, mean |h|, teacher-forcing share, births/deaths, food count.
      Drives all future "did agents get smarter?" claims. `std::format` + `std::ofstream`.

**Acceptance:** metrics dashboard (from telemetry) shows: agents survive longer without
teacher forcing as generations pass; state distribution isn't degenerate (all-ROAM);
no population explosion/crash oscillation at default params.

## 5. Rendering & Main Loop

- [ ] **Instanced rendering for agents and food** in `ls3DGraph` (one VBO of the
      tetrahedron, per-instance transform + color via instanced attribs) instead of
      per-agent draw-call/state churn — big CPU-side win at 1000 agents.
- [ ] Merge static world geometry (grid) into one VAO; avoid re-uploading per frame.
- [ ] Tick/render decoupling: fixed-timestep sim (exists, 33 ms) + render interpolation
      using `m_prev_transform` (already stored! just unused for interpolation).
- [ ] `ls3DGraph` render path reads `m_agents` under `m_data_mutex` — switch to a
      **double-buffered snapshot** (swap a `shared_ptr<const WorldView>`) to remove
      renderer/sim lock contention.

## 6. Hardening & Process

- [ ] Real test suite (ask before adding a framework — GoogleTest or Catch2):
      `lsRNN` forward/backprop numerical-gradients check, mutation bounds, spatial-grid
      query correctness vs brute force, serialize/reproduce determinism with fixed seed.
- [ ] `cppcheck` target already exists — also wire `clang-tidy` config + CI gate
      (GitHub Actions: configure → build → ctest → bench smoke).
- [ ] `.clang-format` (tabs, Allman-ish brace matching current style) and one
      formatting commit, enforced in CI.
- [ ] Consider `vcpkg`/system-package docs update in Readme for new deps (none planned —
      C++23 + existing stack only).

---

## Explicit Non-Goals (this round)

- No new third-party dependencies (no Eigen, no EnTT) — stdlib + GLM + gtkmm only.
- No audio (OpenAL stays unused).
- No GPU-compute NN inference (SIMD is the ceiling for now).
- No multiplayer/networking.

## Risk Register

| Risk | Mitigation |
|------|-----------|
| float conversion changes evolution dynamics | Keep a `GOLNN_USE_DOUBLE` CMake option; A/B with seeded runs |
| RNG stream reordering breaks reproducibility | Central seeded engine; record seed in telemetry header |
| Teacher-forcing anneal causes mass extinction | Slow decay + fallback: if population < threshold, boost regrowth & reset anneal floor |
| Multithreading introduces data races | ASan/TSan preset; interactions phase stays single-threaded |
| `vector<Agent>` pointer invalidation bites | Milestone 2 fixes storage policy first |
