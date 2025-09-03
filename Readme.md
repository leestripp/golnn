# golnn v0.9.0

This project is an Artificial Life (A-Life) simulation built in C++20 and OpenGL. It features a 3D world where autonomous agents, controlled by evolving Recurrent Neural Networks (RNNs), compete for resources, reproduce, and adapt over generations. The goal is to observe emergent complex behaviors, such as flocking, pursuit-evasion, and population dynamics, from a set of simple rules and the process of natural selection.

The application uses `gtkmm-4.0` for the windowing and UI components, and a custom OpenGL widget for rendering the 3D scene.

## Key Features

*   **3D Environment:** A bounded 3D world rendered with modern OpenGL and GLSL shaders.
*   **Evolving Agents:** Agents' "brains" are neural networks that are passed down to offspring with slight mutations, allowing for evolution over time.
*   **Dual-Brain Architecture:** Each agent has two distinct RNNs for decision-making:
    *   `ls3DRNN`: A custom RNN that controls 3D movement and orientation. Its hidden state is the agent's 4x4 transformation matrix.
    *   `lsRNN`: A standard RNN that selects the agent's high-level behavioral state (e.g., `HUNT`, `FLEE`, `ROAM`).
*   **Diverse Agent Types & Ecosystem:**
    *   **Feeders:** Blue agents that seek out plant-based food and flee from predators.
    *   **Predators:** Yellow agents that hunt Feeders. If no Feeders are available, they will prey on each other.
*   **Complex Sensory Inputs:** Agents perceive their environment through a variety of sensors, providing input to their brains:
    *   Direction and distance to food/prey/threats.
    *   A boundary avoidance vector to prevent getting stuck on walls.
    *   Local agent density (friendly vs. hostile) to enable social behaviors.
*   **Teacher-Forced Learning:** Newly spawned agents are initially guided by a "teacher" algorithm that demonstrates correct behavior (e.g., move towards food when hungry). This kickstarts the learning process, which is then refined by evolution.

## Technology Stack

*   **Language:** C++20
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

## Controls

*   **`W, A, S, D`**: Move the camera forward, left, backward, and right.
*   **`Left Shift`**: Sprint (moves the camera faster).
*   **`Left Mouse Button + Drag`**: Look around and orient the camera.
*   **`E`**: Spawn 20 new agents into the world.
*   **`R`**: Spawn 100 new food items into the world.
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

