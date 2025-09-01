# golnn v0.8.0

[![Watch the Video](https://www.youtube.com/embed/T0ZFqmc2e38?si=MCB73gAaJPo5LG32)](https://www.youtube.com/watch?v=T0ZFqmc2e38&t=604s)

This project is an Artificial Life simulation built in C++ and OpenGL. It features autonomous agents, powered by Recurrent Neural Networks (RNNs), that live, eat, reproduce, and evolve in a 3D environment. The simulation uses a dual-network architecture to create complex and emergent behaviors. The user interface and windowing are handled by gtkmm4.

## Core Features

-   **3D Environment:** A fully 3D world rendered with modern OpenGL.
-   **Autonomous Agents:** The world is populated by agents whose actions are determined entirely by their neural network "brains".
-   **Evolutionary Algorithm:** Agents that successfully gather energy can reproduce, passing their neural network weights to their offspring with a chance of mutation. This allows for the evolution of successful strategies over time.
-   **Diverse Population:** New agents are spawned with randomized brain architectures (varying numbers of hidden layers and neurons), creating a diverse gene pool for evolution to act upon.
-   **Two Agent Types:**
    -   **Feeders (Blue):** These agents seek out food, flee from predators, and try to survive.
    -   **Predators (Yellow):** These agents hunt Feeders to gain energy. If no Feeders are available, they will resort to cannibalism.
-   **Dual-Network AI Architecture:** Each agent uses two distinct neural networks that work in concert to make decisions.
-   **Interactive Camera:** A fly-through camera allows you to observe the simulation from any angle.

## The Dual-Network AI Architecture

A key feature of this simulation is the separation of high-level strategy from low-level motor control. Each agent has two brains that work together.

### 1. `lsRNN` - The State Selection Network

This is the agent's "strategic brain." It is a simple Recurrent Neural Network responsible for deciding the agent's current high-level **state**.

-   **Purpose:** To choose one of four states: `IDLE`, `ROAM`, `HUNT`, or `FLEE`.
-   **Inputs:** It processes a simplified, abstract view of the world:
    -   Current Energy Level (quantized into bins).
    -   Distance to the nearest Threat (enemy agent, binned).
    -   Distance to the nearest Food source (binned).
-   **Output:** A classification that determines the agent's current state.
-   **Training:** This network is trained using "teacher-forcing." A set of hardcoded rules determines the "correct" state for any given situation (e.g., if a predator is near and energy is high, the correct state is `FLEE`). The `lsRNN` is trained at every step to mimic this "teacher," allowing it to quickly learn the fundamental logic of survival.

### 2. `ls3DRNN` - The Movement Control Network

This is the agent's "motor cortex." It takes the high-level state from the `lsRNN` and turns it into concrete actions in the 3D world.

-   **Purpose:** To generate the precise translation and rotation needed to move the agent.
-   **Inputs:** It receives a detailed "perception vector" containing:
    -   The world position (x, y, z) and distance to the nearest food.
    -   The world position and distance to a nearby friendly agent to follow.
    -   The world position and distance to a threat or prey.
    -   The agent's own energy level.
    -   **Crucially, it also receives the current state (e.g., `HUNT`) from the `lsRNN` as a one-hot encoded vector.**
-   **Output:** A 7-dimensional vector that defines a delta-transform: 3 values for translation and 4 values for a rotation quaternion.
-   **Training:** This network is also trained via teacher-forcing. Based on the current state, an "ideal" movement is calculated (e.g., in the `HUNT` state, the ideal movement is towards the target). The `ls3DRNN` is then trained to produce the output that achieves this ideal movement.

### Synergy

The two networks create a powerful synergy. The `lsRNN` decides **what to do**, and the `ls3DRNN` figures out **how to do it**. By feeding the state into the movement brain, we provide essential context. The `ls3DRNN` learns different movement patterns for different states, leading to more intelligent and nuanced behavior. For example, it learns to move towards a target when hunting, but away from the *same* target when fleeing.

## Evolution and Learning

-   **Randomized Brains:** When agents are first spawned, their `ls3DRNN` brains are given a random number of hidden layers and neurons (up to a defined maximum). This creates variety in the initial population.
-   **Reproduction & Mutation:** When an agent gathers enough energy, it reproduces. The child inherits a copy of the parent's `ls3DRNN`, but with small, random mutations applied to its weights and biases. This is the core mechanism of evolution.
-   **Alpha Agent Tracking:** When the simulation ends because the last agent has died, the console will output the brain architecture of this final "alpha" agent. This helps track which brain structures were the most successful survivors.

## Building and Running

### Dependencies

The project uses CMake to manage dependencies. You will need the development packages for:
-   `gtkmm-4.0`
-   `sigc++-3.0`
-   `GLEW`
-   `OpenGL`
-   `glm`
-   `OpenAL` - for later.

On a Debian-based system (like Ubuntu), you can install them with:
```bash
sudo apt-get install build-essential cmake libgtkmm-4.0-dev libglew-dev libglm-dev libopenal-dev
```

### Build Instructions

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-directory>

# Configure and build with CMake
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Running the Simulation

The executable will be located in the `build/GameOfLifeNN` directory.
```bash
./GameOfLifeNN/golnn
```

## Controls

-   **`W, A, S, D`**: Move the camera forward, left, back, and right.
-   **`Shift`**: Hold to move the camera faster.
-   **`LMB + Drag`**: Click and drag the left mouse button to look around.
-   **`E`**: Spawn 20 new agents at random locations.
-   **`R`**: Spawn 100 new food items.
-   **`Esc`**: Close the application.

