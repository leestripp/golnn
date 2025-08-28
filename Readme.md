# golnn v0.7.0

[![Watch the Video](https://www.youtube.com/embed/T0ZFqmc2e38?si=MCB73gAaJPo5LG32)](https://www.youtube.com/watch?v=T0ZFqmc2e38&t=604s)

**Evolving AI Agents in a 3D World | Artificial Life Simulation in C++ & OpenGL**

This project is a C++ and OpenGL-based artificial life simulation where autonomous agents evolve complex behaviors in a 3D environment. Each agent is controlled by its own unique Recurrent Neural Network (RNN) that mutates and is passed down through generations. The simulation provides a sandbox for exploring emergent behaviors as agents compete for resources, evade threats, and reproduce.

## Features

### Evolving Neural Networks

*   **Randomized Brain Structure:** New agents are spawned with a randomized number of hidden layers and neurons per layer (up to a configured maximum), promoting architectural diversity from the start.
*   **Genetic Inheritance & Mutation:** When an agent reproduces, its offspring inherits the parent's RNN weights and structure, with a small chance of mutation. This allows for the gradual evolution of behaviors over generations.
*   **Evolving Learning Rates:** Each agent has its own learning rate (between `0.0001` and `0.01`), which also mutates upon reproduction. This allows the simulation to discover optimal learning speeds for different strategies.

### Complex Agent Behaviors

*   **Two Agent Types:**
    *   **Feeders (Blue):** These agents seek out food to gain energy. They will actively flee from nearby Predators.
    *   **Predators (Yellow):** These agents hunt Feeders to drain their energy. If no Feeders are available, they will resort to cannibalism and hunt other Predators.
*   **State-Driven AI:** Agents operate on a simple state machine (Idle, Hunt, Flee, Roam) which guides their high-level goals. The RNN is then trained in real-time to execute the actions required to fulfill these goals.
*   **Vector-Based Perception:** Agents perceive their world not in absolute coordinates, but in terms of a **direction vector and distance** to targets (food, threats, other agents). This makes their learned behaviors more robust and independent of their absolute position in the world.

### Dynamic Simulation Environment

*   **Energy Dynamics:** Agents consume energy over time and must eat to replenish it. Energy levels directly influence an agent's color—brighter means higher energy, darker means lower energy.
*   **Reproduction & Population Control:** Agents with sufficient energy will reproduce, splitting their energy with their offspring. To prevent population explosions, agents stop actively hunting for food once they have enough energy to reproduce, and a global agent limit is enforced.
*   **Alpha Agent Tracking:** When the simulation ends (i.e., the last agent dies), the console outputs the characteristics of this final "Alpha Agent"—including its type, hidden layer count, and neurons per layer. This provides valuable insight into which traits were most successful.

### Interactive 3D Visualization

*   Built with modern OpenGL and a `gtkmm-4.0` window.
*   A free-look camera allows you to observe the simulation from any angle.
*   Runtime controls to dynamically spawn new agents and food.

## Core Technologies

*   **Language:** C++20
*   **Graphics:** OpenGL 3.3+
*   **Windowing/UI:** gtkmm-4.0
*   **Math:** GLM (OpenGL Mathematics)
*   **OpenGL Loading:** GLEW
*   **Build System:** CMake

## Building the Project

This project uses CMake. Ensure you have the required dependencies installed on your Linux system.

### Dependencies (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config libgtkmm-4.0-dev libglew-dev libglm-dev libopenal-dev cppcheck
```

### Build Steps

```bash
# 1. Clone the repository
git clone <gitlab-repo-url>
cd <project-directory>

# 2. Create and enter a build directory
mkdir build && cd build

# 3. Configure the project with CMake
cmake ..

# 4. Compile the project (using all available CPU cores)
make -j$(nproc)

# 5. (Optional) Run static analysis with cppcheck
make cppcheck
```

## Running the Simulation

After a successful build, the executable will be located in the `build/GameOfLifeNN` directory.

```bash
cd build/GameOfLifeNN
./golnn
```

## Controls

*   **Move Camera:** `W`, `A`, `S`, `D`
*   **Sprint Camera:** Hold `Left Shift` while moving
*   **Look Around:** Hold `Left Mouse Button` and drag the mouse
*   **Spawn New Agents:** Press `E`
*   **Spawn New Food:** Press `R`
*   **Quit:** Press `Esc`
