<!-- monkeybyte:type: cpp -->

# AGENTS.md — GameOfLifeNN

Guidance for AI coding agents working in this repository.

## Project Overview

**GameOfLifeNN** (binary: `golnn`, v0.9.0) is an Artificial Life (A-Life) simulation
written in C++20. Autonomous 3D agents with evolving Recurrent Neural Network (RNN)
"brains" compete for food, reproduce with mutation, and adapt over generations in a
bounded OpenGL-rendered world. The goal is emergent behavior (flocking, pursuit-evasion,
population dynamics) from simple rules plus natural selection.

Each agent has a **dual-brain architecture**:

- `ls3DRNN` — custom RNN controlling 3D movement/orientation; its hidden state is the
  agent's 4x4 transformation matrix. Teacher-forced on *direction vectors* from sensory
  input (never absolute positions).
- `lsRNN` — standard RNN that selects the high-level behavioral state (`IDLE`, `ROAM`,
  `HUNT`, `FLEE`) from inputs like energy level and distances to food/enemies.

Agent types: **Feeders** (blue, eat vegetation, flee predators) and **Predators**
(yellow, hunt feeders; fallback to meat, then each other). Dead feeders drop `MEAT`
food inside world bounds. Vegetation regrows on a timer.

## Tech Stack

| Area      | Technology                                   |
|-----------|----------------------------------------------|
| Language  | C++20 (`CMAKE_CXX_STANDARD 20`, required)    |
| GUI       | gtkmm-4.0 (windowing, event controllers)     |
| Graphics  | OpenGL + GLEW, custom GTK OpenGL widget      |
| Math      | GLM (header-only)                            |
| Signals   | libsigc++-3.0                                |
| Audio     | OpenAL (found by CMake, currently unused in sources) |
| Build     | CMake (min 3.28) + pkg-config                |
| Static analysis | cppcheck (custom `cppcheck` target)      |

## Dependencies (Debian/Ubuntu)

```bash
sudo apt-get install build-essential cmake pkg-config cppcheck \
     libgtkmm-4.0-dev libsigc++-3.0-dev libglew-dev libglm-dev libopenal-dev
```

## Build, Run, Verify

```bash
# Configure & build (CMakePresets.json: default/debug/release/asan/ubsan)
cmake --preset default
cmake --build --preset default -j

# Headless benchmark — no display needed: ticks agents food world
./build/golnn_bench 300 1000 500 100

# Smoke test (CTest): constructs a Simulation and ticks it 100x
ctest --test-dir build

# Sanitizer builds run the same test
cmake --preset asan && cmake --build --preset asan && ctest --test-dir build-asan

# Run the GUI simulation (requires a display / GL context)
./build/GameOfLifeNN/golnn

# Static analysis (custom target; CPPCHECK_EXITCODE=2 set on the target)
cmake --build --target cppcheck
```

- Default preset is `Release` (`-O3`); `Debug` adds `-g`; `asan`/`ubsan` enable
  the sanitizers for the core library, bench, and test.
- Verification = warning-clean build (warnings are errors in spirit) + the
  full CTest suite (`smoke` + per-component unit tests) + `golnn_bench` A/B
  for performance work.
- `cppcheck` is a `REQUIRED` find_program at configure time — configure fails without it.

## CMake Targets

- `golnn_core` — static library with the full simulation (`GameOfLifeNN/src/Agent.cpp`,
  `Simulation.cpp`, `SpatialGrid.cpp`, `lsRNN/`, `ls3DRNN/`). No GUI deps; C++23.
- `golnn_bench` — headless benchmark executable (repo-root `bench/bench.cpp`),
  links only `golnn_core`. Run: `./build/golnn_bench <ticks> <agents> <food> <world>`.
- `golnn` — the GTK GUI executable (same sources as before + `main.cpp`, ls3DGraph,
  ls3DNNHelpers, lsStatistics), built only when gtkmm/OpenGL are found
  (`GameOfLifeNN/CMakeLists.txt`). Links: `${GTKMM_LIBRARIES}`, `${SIGC_LIBRARIES}`,
  `GLEW::glew`, `OpenGL::GL`.
- `golnn_smoke_test` — CTest-registered smoke test (repo-root `test/smoke_test.cpp`).
- `golnn_test_spatial_grid`, `golnn_test_lsrnn`, `golnn_test_ls3drnn`,
  `golnn_test_lsvec_tokenizer` — unit tests (repo-root `test/`), each a separate
  CTest entry. They use the framework-free harness in `test/test_framework.h`
  (`TEST_CASE`/`CHECK`/`CHECK_NEAR`/`TEST_MAIN`); add new component tests the
  same way and register them with `add_test`.
- `golnn-warnings` — INTERFACE target carrying the extended warning flags, linked
  into `golnn_core`/`golnn_bench` (GUI target pending a warning-clean pass).
- `cppcheck` — custom static-analysis target (root `CMakeLists.txt`).
- Sources are listed explicitly per target (no GLOB) — new files must be added to
  `coreSources` / the GUI target in `GameOfLifeNN/CMakeLists.txt`.

## Project Structure

```
CMakeLists.txt            # Root: deps, flags, cppcheck target, add_subdirectory
GameOfLifeNN/
  CMakeLists.txt          # Defines the golnn executable
  src/
    main.cpp              # Entry point: GTK app/window, main loop, camera controls
    Agent.h / Agent.cpp   # Agent attributes, brains, perceive-think-act cycle
    Simulation.h/.cpp     # World state, spawning, interactions, regrow timer
    SpatialGrid.h/.cpp    # Uniform spatial hash grid for O(k) proximity queries
    bench/bench.cpp       # Headless benchmark: ticks/sec at N agents (repo root)
    test/                 # CTest smoke test + per-component unit tests
    ls3DGraph/            # Custom GTK OpenGL widget (3D scene rendering)
    ls3DRNN/              # Custom movement RNN (transform-matrix hidden state)
    lsRNN/                # State-selection RNN + lsVecTokenizer helper
Readme.md                 # User-facing docs, controls, feature list
icon/                     # App icon
build/                    # Out-of-source build dir (gitignored)
```

Runtime controls: `WASD` camera, `Shift` sprint, mouse-drag look, `E` spawn 20 agents,
`R` spawn 100 food, `Esc` quit. Tick rate ~33 ms (`TICK_RATE_MS` in `main.cpp`).

## Code Style & Conventions

- **Indentation: tabs** (existing files mix some spaces — match surrounding code).
- No `.clang-format` / `.editorconfig`; the existing sources are the style reference.
- Member variables use `m_` prefix (`m_energy`, `m_brain`); class names are
  `PascalCase` for framework-level classes and carry an `ls` prefix for the project's
  custom types (`lsRNN`, `ls3DRNN`, `ls3DGraph`, `lsVecTokenizer`).
- **Headers use old-style include guards** (`#ifndef AGENT_H` / `#define` / `#endif`) —
  do not switch to `#pragma once` for consistency.
- Legacy `using namespace std;` appears in headers (`Agent.h`, `lsRNN.h`) and code is
  written accordingly. Match this in existing files; prefer explicit `std::` in **new**
  headers where it doesn't clash with surrounding style.
- Enums are `enum class` with `UPPER_SNAKE` values (`AgentType::FEEDER`, `State::HUNT`).
- Randomness: `std::random_device` + `std::mt19937` statics (see `Agent.cpp`), not `rand()`.
- Config limits are `static constexpr` members on `Simulation`
  (e.g. `MAX_AGENTS = 1000`, `MAX_LAYER_COUNT = 10`).
- Warning flags: `-Wall -Wextra` globally. **Keep the build warning-clean**; warnings
  are errors in spirit.
- Comments are plentiful explanatory `//` comments — follow that habit for
  non-obvious simulation logic.

## Conventions & Gotchas for Agents

- Teacher forcing and reproduction/mutation logic are the simulation's core —
  the rules (eating order, detection distances, direction-vs-position teaching,
  population caps) are documented in the code comments and Readme.md; read them
  before modifying that logic.
- Guard population explosions: check `MAX_AGENTS` before spawning/reproducing.
- Any position dropped into the world (e.g. `MEAT` food) must be clamped inside world
  bounds — agents can die outside them.
- The simulation runs on a GLib timeout in the GTK main loop; keep per-tick work
  cheap (1000 agents × per-tick ticks at ~30 fps).
- Hot-path rules (post-M1/M2, see Plan.md): all NN math is `float` in flat
  contiguous arrays; `perceive()`/`train_state_brain()`/NN forward passes reuse
  member scratch buffers (no per-tick heap allocations); `Simulation` rebuilds
  its four type-partitioned `SpatialGrid`s each tick and all proximity queries
  go through them (never linear scans of `m_agents`/`m_food`). If you add a new
  proximity query, add it to the grids — don't reintroduce an O(n) scan.
- `ls3DRNN::forward()` is memoized (Agent calls train_step then process_step with
  the same input each tick). Any change that mutates weights must invalidate the
  cache — `optimizer_step()` and `mutate()` already do; keep that invariant.
- **Wayland + GLEW gotcha**: under Wayland, GTK4's GLArea hands GLEW an EGL
  context, so `glewInit()` returns `GLEW_ERROR_NO_GLX_DISPLAY` (4). GLEW loads
  all *core* GL function pointers before that GLX probe fails, so the error is
  harmless — `ls3DGraph::on_realize()` accepts error 4 but still verifies the
  core entry points (`glCreateShader`/`glGenVertexArrays`/`glBufferData`/
  `glUseProgram`) are non-null before continuing. Don't "fix" this by
  reverting to a hard `!= GLEW_OK` abort, and don't try `GDK_BACKEND=x11` —
  the X11 backend exits silently with no window on this machine.
- `OpenAL` is found by CMake but not linked/used — don't add audio dependencies
  without asking.
- Do not commit anything from `build/`; it is gitignored.
- Ask before adding dependencies or `FetchContent` entries; prefer the standard
  library and existing deps (GLM, sigc++, gtkmm).
