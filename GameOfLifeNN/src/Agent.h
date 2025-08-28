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
    // Define agent types
    enum class AgentType {
        FEEDER,
        PREDATOR
    };
	
    // Define agent states
    enum class State {
		IDLE,
		ROAM,
		HUNT,
		FLEE
    };
	
	Agent(ls3DRNN brain, const glm::mat4& initial_transform, AgentType type, float learning_rate = -1.0f, float initial_energy = -1.0f);
 	
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
    glm::vec3 m_color; // Final color for visualization, adjusted for energy
	AgentType m_type;  // The type of agent
 	
private:
    void update_state_transitions(const Simulation& world);
    void train_brain(const Simulation& world, const std::vector<float>& perception_vector);
    std::vector<float> perceive(const Simulation& world) const;

    // These are the target positions for the RNN inputs, updated each tick.
    // They are set to zero if no target is available in the current state.
	glm::vec3 m_food_pos;
	glm::vec3 m_follow_agent_pos; // For roaming with other feeders
	glm::vec3 m_threat_pos;       // Position of predator to flee from, or prey to hunt
	
    glm::vec3 m_base_color;      // Base color determined by agent type
    float m_energy_decay_rate; // Agent's specific energy decay per tick
    float m_learning_rate;     // Each agent has its own learning rate

    // Simulation Parameters
    static constexpr float INITIAL_ENERGY = 100.0f;
    static constexpr int MAX_AGE = 5000;
    static constexpr float MAX_SPEED = 0.75f;
	static constexpr float ROTATION_LERP_FACTOR = 0.25f;
	
	static constexpr float ROTATION_WEIGHT = 0.25f;
	
    // --- Mutation Parameters ---
    static constexpr float MUTATION_RATE = 0.05f;   // 5% chance for a weight/bias to mutate
    static constexpr float MUTATION_AMOUNT = 0.01f; // Max change during mutation
	
    // Agent Type Specific Parameters
    static constexpr float FEEDER_ENERGY_DECAY = 0.1f;
    static constexpr float PREDATOR_ENERGY_DECAY = 0.15f; // Predators die off to quick, their food is moving.
	
    static constexpr float FEEDER_ENERGY_TO_REPRODUCE = 140.0f;
    static constexpr float PREDATOR_ENERGY_TO_REPRODUCE = 155.0f;
	
    // State transition parameters
    static constexpr float ROAM_FOLLOW_RADIUS_SQ = 50.0f;			// ~7 units squared
	static constexpr float PREDATOR_DETECTION_RADIUS_SQ = 100.0f;	// 10 units squared
    static constexpr float PREDATOR_HUNT_RADIUS_SQ = 225.0f;        // 15 units squared
};

#endif // AGENT_H
