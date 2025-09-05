#ifndef AGENT_H
#define AGENT_H

#include "ls3DRNN.h"
#include "lsRNN.h"
#include "lsVecTokenizer.h"
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
	
	Agent(ls3DRNN brain, const glm::mat4& initial_transform, AgentType type, float learning_rate = -1.0f, float initial_energy = -1.0f, size_t state_hidden_dim = 0);
 	
	void update(const Simulation& world);
	
	bool isDead() const;
	bool canReproduce() const;
	Agent reproduce();
	
	float get_learning_rate() const;
	size_t get_state_brain_hidden_dim() const;
	
	// -- Public State --
	ls3DRNN m_brain;
	glm::mat4 m_transform;
	glm::mat4 m_prev_transform;
	float m_energy;
	int m_age;
	State m_state;	 // The current state of the agent
	glm::vec3 m_color; // Final color for visualization, adjusted for energy
	AgentType m_type;  // The type of agent
 	
	// The size of the perception vector fed into the movement brain (ls3DRNN).
	// [0-11] Visual (Food, Threat, Friend), [12-19] Non-Visual (Scent, Sound)
	// [20-22] Boundary, [23-24] Density, [25-26] Internal (Energy, Age)
	// [27-30] State (one-hot)
	static constexpr size_t PERCEPTION_VECTOR_SIZE = 31;

	// State Brain Parameters
	static constexpr size_t NUM_ENERGY_BINS = 4;
	static constexpr size_t NUM_DISTANCE_BINS = 4;		// For enemy distance
	static constexpr size_t NUM_FOOD_DISTANCE_BINS = 4;   // For food distance
	static constexpr size_t STATE_BRAIN_INPUT_DIM = NUM_ENERGY_BINS + NUM_DISTANCE_BINS + NUM_FOOD_DISTANCE_BINS;
	static constexpr size_t NUM_STATES = 4; // IDLE, ROAM, HUNT, FLEE
	// State brain hidden layer size is now randomized on creation
	static constexpr size_t MIN_STATE_NEURONS = 4;
	static constexpr size_t MAX_STATE_NEURONS = 32;
	
private:
	void update_state_transitions(const Simulation& world);
	void train_brain(const Simulation& world, const std::vector<float>& perception_vector);
	void train_state_brain(const Simulation& world);
	std::vector<float> perceive(const Simulation& world);
	glm::vec3 find_closest_food_direction(const Simulation& world, size_t food_type) const;

	// These are the target positions for the RNN inputs, updated each tick.
	// They are set to zero if no target is available in the current state.
	glm::vec3 m_food_pos;
	glm::vec3 m_follow_agent_pos; // For roaming with other feeders
	glm::vec3 m_threat_pos;	   // Position of predator to flee from, or prey to hunt
	
	glm::vec3 m_base_color;	  // Base color determined by agent type
	float m_energy_decay_rate; // Agent's specific energy decay per tick
	float m_learning_rate;	 // Each agent has its own learning rate
	size_t m_state_brain_hidden_dim; // Number of neurons in the state brain's hidden layer
	lsRNN m_state_brain;	   // New state-selection brain
	
	// batch learning
	size_t m_gradient_accumulation_step;
	static constexpr size_t BATCH_SIZE = 30; // Collect gradients for 30 ticks before updating
	
	// Simulation Parameters
	static constexpr float INITIAL_ENERGY = 100.0f;
	static constexpr int MAX_AGE = 5000;
	static constexpr float MAX_SPEED = 1.0f;
	static constexpr float ROTATION_LERP_FACTOR = 0.25f;
	
	static constexpr float ROTATION_WEIGHT = 0.25f;
	
	// --- Mutation Parameters ---
	static constexpr float MUTATION_RATE = 0.02f;   // 2% chance for a weight/bias to mutate
	static constexpr float MUTATION_AMOUNT = 0.01f; // Max change during mutation
	
	// Agent Type Specific Parameters
	static constexpr float FEEDER_ENERGY_DECAY = 0.01f;
	static constexpr float PREDATOR_ENERGY_DECAY = 0.03f;
	
	static constexpr float FEEDER_ENERGY_TO_REPRODUCE = 130.0f;
	static constexpr float PREDATOR_ENERGY_TO_REPRODUCE = 150.0f;
	
	// State transition parameters
	static constexpr float ROAM_FOLLOW_RADIUS_SQ = 100.0f;			// units squared
	static constexpr float PREDATOR_DETECTION_RADIUS_SQ = 400.0f;	// 20 units squared
	static constexpr float FEEDER_FOOD_DETECTION_RADIUS_SQ = 400.0f; // 20 units squared
	static constexpr float PREDATOR_HUNT_RADIUS_SQ = 400.0f;		// 20 units squared

	// New Sensor Parameters
	static constexpr float BOUNDARY_THRESHOLD = 5.0f;
	static constexpr float DENSITY_RADIUS_SQ = 25.0f; // 5 units squared
};

#endif // AGENT_H
