#ifndef AGENT_H
#define AGENT_H

#include "ls3DRNN.h"
#include "lsRNN.h"
#include "lsVecTokenizer.h"
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <random>

#include "StateChecksum.h"
#include "BrainCatalog.h"
#include "Perception.h"

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
	
	Agent(ls3DRNN brain, const glm::mat4& initial_transform, AgentType type, float learning_rate = -1.0f, float initial_energy = -1.0f, size_t state_hidden_dim = 0, std::uint32_t random_seed = std::random_device{}(), BrainTier brain_tier = BrainTier::CUSTOM);
 	
	void update(const Simulation& world);
	bool prepare_update(const Simulation& world);
	void train_movement();
	void finish_update(const Simulation& world);
	bool movement_pending() const { return m_movement_pending; }
	BrainTier brain_tier() const { return m_brain_tier; }
	static constexpr float movement_rotation_weight() { return ROTATION_WEIGHT; }
	const std::vector<float>& movement_target() const { return m_movement_target; }
	
	bool isDead() const;
	bool canReproduce() const;
	Agent reproduce();
	
	float get_learning_rate() const;
	size_t get_state_brain_hidden_dim() const;
	state_checksum::type checksum() const;
	
	// -- Public State --
	ls3DRNN m_brain;
	glm::mat4 m_transform;
	glm::mat4 m_prev_transform;
	glm::vec3 m_velocity{0.0f}; // Last accepted world-space movement / MAX_SPEED
	float m_energy;
	int m_age;
	State m_state;	 // The current state of the agent
	glm::vec3 m_color; // Final color for visualization, adjusted for energy
	AgentType m_type;  // The type of agent
 	
	// Versioned layout is documented in Perception.h. v2 preserves v1 channels
	// and appends motion, goal, energy-trend, density, and proximity signals.
	static constexpr size_t PERCEPTION_VECTOR_SIZE = perception::COUNT;

	// State Brain Parameters
	static constexpr size_t NUM_ENERGY_BINS = 4;
	static constexpr size_t NUM_DISTANCE_BINS = 4;		// For enemy distance
	static constexpr size_t NUM_FOOD_DISTANCE_BINS = 4;   // For food distance
	static constexpr size_t STATE_BRAIN_INPUT_DIM = perception::STATE_INPUT_COUNT;
	static constexpr size_t NUM_STATES = 4; // IDLE, ROAM, HUNT, FLEE
	// State brain hidden layer size is now randomized on creation
	static constexpr size_t MIN_STATE_NEURONS = 4;
	static constexpr size_t MAX_STATE_NEURONS = 32;
	
	std::vector<float> m_perception; // scratch, reused every tick by perceive()
	std::vector<float> m_state_input_scratch; // scratch, reused every tick by train_state_brain()

private:
	void update_state_transitions(const Simulation& world);
	void prepare_brain_target(std::vector<float>& perception_vector);
	void train_state_brain(const Simulation& world);
	// Fills and returns the reused m_perception scratch buffer. update() writes
	// the state one-hot indices [27-30] into it after calling this.
	std::vector<float>& perceive(const Simulation& world);

	// These are the target positions for the RNN inputs, updated each tick.
	// They are set to zero if no target is available in the current state.
	glm::vec3 m_food_pos;
	glm::vec3 m_follow_agent_pos; // For roaming with other feeders
	glm::vec3 m_threat_pos;	   // Position of predator to flee from, or prey to hunt
	BrainTier m_brain_tier;
	bool m_movement_pending = false;
	std::vector<float> m_movement_target;
	
	glm::vec3 m_base_color;	  // Base color determined by agent type
	float m_energy_decay_rate; // Agent's specific energy decay per tick
	float m_energy_trend = 0.0f; // Smoothed energy delta, normalized in perception
	float m_previous_tick_energy = 0.0f;
	float m_learning_rate;	 // Each agent has its own learning rate
	std::mt19937 m_rng; // Per-agent stream: safe for parallel updates and reproduction
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
	static constexpr float FEEDER_ENERGY_DECAY = 0.02f;
	static constexpr float PREDATOR_ENERGY_DECAY = 0.05f;
	
	static constexpr float FEEDER_ENERGY_TO_REPRODUCE = 110.0f;
	static constexpr float PREDATOR_ENERGY_TO_REPRODUCE = 130.0f;
	
	// State transition parameters
	static constexpr float ROAM_FOLLOW_RADIUS_SQ = 100.0f; // units squared
	
	// New Sensor Parameters
	static constexpr float BOUNDARY_THRESHOLD = 1.0f;
	static constexpr float DENSITY_RADIUS_SQ = 1.0f;
	static constexpr float FOOD_DENSITY_RADIUS_SQ = 100.0f;
	static constexpr float BOUNDARY_AWARENESS_DISTANCE = 10.0f;
};

#endif // AGENT_H
