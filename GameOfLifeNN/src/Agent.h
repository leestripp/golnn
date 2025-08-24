#ifndef AGENT_H
#define AGENT_H

#include "ls3DRNN.h"
#include <glm/glm.hpp>
#include <vector>

// Forward declaration
class Simulation;

using namespace std;

class Agent
{
public:
    // Define agent states
    enum class State {
        IDLE,
        ROAM,
        HUNT
    };

    Agent(ls3DRNN brain, const glm::mat4& initial_transform);
	
    void update(const Simulation& world);
	
    bool isDead() const;
    bool canReproduce() const;
    Agent reproduce();
	
    // -- Public State --
    ls3DRNN m_brain;
    glm::mat4 m_transform;
    glm::mat4 m_prev_transform;
    float m_energy;
    int m_age;
    State m_state;     // The current state of the agent
    glm::vec3 m_color; // Color for visualization
	
private:
    void update_state_transitions(const Simulation& world);
    void train_brain(const Simulation& world, const std::vector<float>& perception_vector);
    std::vector<float> perceive(const Simulation& world) const;

    // These are the target positions for the RNN inputs, updated each tick.
    // They are set to zero if no target is available in the current state.
    glm::vec3 m_food_pos;
    glm::vec3 m_follow_agent_pos;

    static constexpr float INITIAL_ENERGY = 100.0f;
    static constexpr float ENERGY_DECAY_PER_TICK = 0.1f;
    static constexpr int MAX_AGE = 5000;
    static constexpr float MAX_SPEED = 5.0f;
	static constexpr float ROTATION_LERP_FACTOR = 0.25f;
	
	static constexpr float LEARNING_RATE = 0.001f;
	static constexpr float ROTATION_WEIGHT = 0.5f;
	
    // State transition parameters
    static constexpr float ENERGY_TO_REPRODUCE = 110.0f;
    static constexpr float HUNTING_ENERGY_THRESHOLD = 150.0f;
    static constexpr float ROAM_FOLLOW_RADIUS_SQ = 25.0f; // 25 = 5 units squared
};

#endif // AGENT_H
