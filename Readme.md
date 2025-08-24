# golnn v0.5.0

[Watch the Video](https://www.youtube.com/embed/T0ZFqmc2e38?si=MCB73gAaJPo5LG32)

**Evolving AI Agents in a 3D World | Artificial Life Simulation in C++ & OpenGL**

---

## Description

Check out my 3D artificial life simulation, built from scratch in modern C++! In this project, autonomous agents with their own Recurrent Neural Network (RNN) brains learn to survive, hunt for food, and reproduce in a dynamic 3D environment.

In the video, you can see the agents' behaviors change based on their needs. Their color indicates their current state:
🟡 **Yellow (HUNT):** Low on energy, actively seeking the nearest food source.
🔵 **Blue (ROAM):** Following other agents, a social behavior that can lead them to food clusters.
⚪ **Grey (IDLE):** No immediate threats or targets, conserving energy.

**How It Works:**
Each agent's "brain" is a custom Recurrent Neural Network. On every simulation tick, the simulation uses a technique called "Teacher Forcing" to train the agent. It calculates the ideal action based on the agent's state (like moving towards food) and teaches the network to produce that result.

When an agent gathers enough energy, it reproduces, creating an offspring. The child inherits the parent's neural network, but with slight random mutations. Over many generations, these mutations allow the agents to evolve more efficient and interesting survival strategies.

**Key Features:**
*   A 3D simulation world with a fly-through camera (WASD controls).
*   Agents with a full lifecycle: energy decay, eating, reproduction, and death.
*   Each agent is controlled by its own unique Recurrent Neural Network.
*   Real-time training and evolution through mutation.
*   Dynamic agent behaviors and emergent group dynamics.

**Technology Stack:**
*   **Language:** C++20
*   **Graphics:** OpenGL with GLEW
*   **GUI/Windowing:** GTKmm 4
*   **3D Math:** GLM (OpenGL Mathematics)
*   **Build System:** CMake

This project was a fantastic challenge, combining AI concepts like neural networks and evolutionary algorithms with real-time 3D graphics programming. Let me know what you think in the comments!

---
#cpp #ai #neuralnetworks #simulation #artificiallife #opengl #gamedev #programming #evolution
