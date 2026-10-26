# golnn v0.9.0

This project is an Artificial Life (A-Life) simulation built in C++23 and OpenGL. It features a 3D world where autonomous agents, controlled by evolving Recurrent Neural Networks (RNNs), compete for resources, reproduce, and adapt over generations. The goal is to observe emergent complex behaviors, such as flocking, pursuit-evasion, and population dynamics, from a set of simple rules and the process of natural selection.

The application uses `gtkmm-4.0` for the windowing and UI components, and a custom OpenGL widget for rendering the 3D scene.

## Key Features

*   **3D Environment:** A bounded 3D world rendered with modern OpenGL and GLSL shaders.
*   **Evolving Agents:** Agents' "brains" are neural networks that are passed down to offspring with slight mutations, allowing for evolution over time.
*   **Dual-Brain Architecture:** Each agent has two distinct RNNs for decision-making:
    *   `ls3DRNN`: A custom RNN that controls 3D movement and orientation. Its hidden state is the agent's 4x4 transformation matrix.
    *   `lsRNN`: A standard RNN that selects the agent's high-level behavioral state (e.g., `HUNT`, `FLEE`, `ROAM`).
*   **Standardized Brain Tiers:** Small, medium, and large dual-brain architectures support mixed populations, inheritance, and exact scalar/batched execution. Press **1/2/3** to choose the brain size for newly spawned agents.
*   **Parallel Simulation:** A persistent CPU thread pool updates agents against an immutable start-of-tick world snapshot, with deterministic serial/parallel replay.
*   **Vulkan + OpenGL Rendering:** Native Vulkan/Wayland is the default, with depth-tested world-cube edges, food, and interpolated instanced-agent pipelines. OpenGL remains available as an automatic or explicit fallback. Both consume the same immutable float snapshots.
*   **Decoupled Instanced Rendering:** Simulation and GTK run independently. Immutable float snapshots feed one instanced agent draw call, direct food buffers, and smooth previous/current transform interpolation without world-lock contention.
*   **Adaptive Ecology:** Bounded vegetation regrowth responds to feeder population and consumption, with an emergency food floor and optional CSV telemetry. Live population and food statistics are shown in the GUI.
*   **Diverse Agent Types & Ecosystem:**
    *   **Feeders:** Blue agents that seek out plant-based food and flee from predators.
    *   **Predators:** Yellow agents that hunt Feeders. If no Feeders are available, they will prey on each other.
*   **Rich Versioned Senses:** A consolidated sensor pass supplies wide-field vision, omnidirectional scent/hearing, motion and heading, teacher-goal alignment, energy trends, food density, and boundary/threat proximity. The state brain receives a compact normalized subset.
*   **Complex Sensory Inputs:** Agents perceive their environment through a variety of sensors, providing input to their brains:
    *   Direction and distance to food/prey/threats.
    *   A boundary avoidance vector to prevent getting stuck on walls.
    *   Local agent density (friendly vs. hostile) to enable social behaviors.
*   **Teacher-Forced Learning:** Newly spawned agents are initially guided by a "teacher" algorithm that demonstrates correct behavior (e.g., move towards food when hungry). This kickstarts the learning process, which is then refined by evolution.

## Technology Stack

*   **Language:** C++23
*   **Graphics:** OpenGL, GLEW
*   **GUI:** gtkmm-4.0
*   **Math:** GLM (OpenGL Mathematics)
*   **Build System:** CMake

## Dependencies

To build this project, you will need the following libraries installed:

*   `gtkmm-4.0`
*   `sigc++-3.0`
*   `glew`
*   `glm`
*   `opengl`
*   `openal`

On a Debian/Ubuntu-based system, you can install them with:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config libgtkmm-4.0-dev libglew-dev libglm-dev libopenal-dev
```

## Benchmarking and Telemetry

```bash
# Repeatable 1000-agent benchmark using 12 workers
./build/golnn_bench 300 1000 500 100 --samples 3 --seed 42 --workers 12 --brain-tier small

# Compare medium/large brains or the scalar reference kernel
./build/golnn_bench 300 1000 500 100 --brain-tier medium
./build/golnn_bench 300 1000 500 100 --brain-tier large
./build/golnn_bench 300 1000 500 100 --brain-tier small --scalar-brains

# Compare the original fixed regrowth and save ecology telemetry
./build/golnn_bench 3000 300 200 100 --fixed-food --telemetry fixed.csv
./build/golnn_bench 3000 300 200 100 --telemetry adaptive.csv
```

`--workers 1` selects the serial fallback; omitting it uses hardware concurrency.
Repeated samples print median timings and verify identical world checksums.

## Building and Running

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Configure and build with CMake:**
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

3.  **Run the simulation:**
    ```bash
    ./GameOfLifeNN/golnn
    ```

## Renderer Selection

```bash
# Native Vulkan/Wayland (default when available)
./build/GameOfLifeNN/golnn

# Force the existing OpenGL backend
GOLNN_RENDERER=opengl ./build/GameOfLifeNN/golnn

# Start with a larger population
GOLNN_INITIAL_AGENTS=1000 ./build/GameOfLifeNN/golnn
```

Vulkan requires a native Wayland session; other environments fall back to OpenGL.
Shader sources and generated SPIR-V are included. Only developers editing shaders need
`glslang-tools` and `spirv-tools` to run `tools/compile_vulkan_shaders.sh`.

## Controls

*   **`W, A, S, D`**: Move the camera forward, left, backward, and right.
*   **`Left Shift`**: Sprint (moves the camera faster).
*   **`Left Mouse Button + Drag`**: Look around and orient the camera.
*   **`E`**: Spawn 20 new agents into the world.
*   **`R`**: Spawn 100 new food items into the world.
*   **`1`, `2`, `3`**: Select small, medium, or large brains for subsequently spawned agents. Existing agents and their offspring retain their own tier.
*   **`Esc`**: Close the application.

## Project Structure

*   `GameOfLifeNN/`: The main project directory.
    *   `src/`: Contains the core source code.
        *   `Agent.h/.cpp`: Defines agent behavior, attributes, brains, and the perceive-think-act cycle.
        *   `Simulation.h/.cpp`: Manages the world state, entities (agents, food), and their interactions.
        *   `main.cpp`: The application entry point, responsible for setting up the GTK window and the main simulation/render loop.
        *   `ls3DRNN/`: Implementation of the custom 3D movement RNN.
        *   `lsRNN/`: Implementation of the standard state-selection RNN.
        *   `ls3DGraph/`: The custom GTK OpenGL widget for rendering the 3D scene.
*   `CMakeLists.txt`: The main CMake build script.

