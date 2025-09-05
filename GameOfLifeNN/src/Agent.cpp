#include "Agent.h"
#include "Simulation.h"
#include "lsRNN.h"
#include "lsVecTokenizer.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/quaternion.hpp>

#include <iostream>
#include <algorithm>
#include <limits>
#include <random>

// Random number generator for agent-specific properties like learning rate
static std::random_device rd;
static std::mt19937 agent_gen(rd());
static std::uniform_real_distribution<float> lr_dist(0.0001f, 0.1f);
static std::uniform_real_distribution<float> mutation_chance_dist(0.0f, 1.0f);
static std::uniform_int_distribution<size_t> state_neuron_dist(Agent::MIN_STATE_NEURONS, Agent::MAX_STATE_NEURONS);

// Static tokenizer for agent states.
static lsVecTokenizer state_tokenizer(Agent::NUM_STATES);
static bool tokenizer_initialized = false;

static void initialize_tokenizer()
{
	if(! tokenizer_initialized)
	{
		state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::IDLE));
		state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::ROAM));
		state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::HUNT));
		state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::FLEE));
		tokenizer_initialized = true;
	}
}


// Agent constructor
Agent::Agent(ls3DRNN brain, const glm::mat4& initial_transform, AgentType type, float learning_rate, float initial_energy, size_t state_hidden_dim) :
	m_brain(std::move(brain)),
	m_transform(initial_transform),
	m_prev_transform(initial_transform),
	m_energy(initial_energy),
	m_age(0),
	m_state(State::IDLE),
	m_color({0.7f, 0.7f, 0.7f}),
	m_type(type),
	m_food_pos(0.0f),
	m_follow_agent_pos(0.0f),
	m_threat_pos(0.0f),
	m_energy_decay_rate(FEEDER_ENERGY_DECAY),
	m_state_brain_hidden_dim(state_hidden_dim == 0 ? state_neuron_dist(agent_gen) : state_hidden_dim),
	m_state_brain(STATE_BRAIN_INPUT_DIM, m_state_brain_hidden_dim, NUM_STATES, Task::CLASSIFICATION),
	m_gradient_accumulation_step(0)
{
	// Ensure tokenizer is set up
	initialize_tokenizer();
	
	// Assign learning rate: random if not provided, otherwise use the given value.
	if(learning_rate < 0.0f)
	{
		m_learning_rate = lr_dist(agent_gen);
	} else
	{
		m_learning_rate = learning_rate;
	}
	
	// If initial_energy was not provided (i.e., it's a newly spawned agent, not an offspring),
	// set the energy based on the agent's type.
	if(m_energy < 0.0f)
	{
		if(m_type == AgentType::PREDATOR)
		{
			m_energy = INITIAL_ENERGY * 1.5f;
		} else
		{
			m_energy = INITIAL_ENERGY;
		}
	}
	
	// Synchronize the brain's internal state with the agent's initial transform.
	m_brain.set_state(m_transform);
	
	// Set initial base color and metabolism based on type
	if(m_type == AgentType::PREDATOR)
	{
		m_base_color = {1.0f, 1.0f, 0.0f}; // Yellow
		m_energy_decay_rate = PREDATOR_ENERGY_DECAY;
	} else
	{
		m_base_color = {0.0f, 0.0f, 1.0f}; // Deep Blue
		m_energy_decay_rate = FEEDER_ENERGY_DECAY;
	}
}

// Main update loop for the agent
void Agent::update(const Simulation& world)
{
	// 1. Basic housekeeping: age and decay energy
	m_age++;
	m_energy -= m_energy_decay_rate;
	if(isDead()) return;
	
	m_prev_transform = m_transform;
	
	// 2. Perceive the world, build the movement brain input vector, and update internal world model
	std::vector<float> perception_vector = perceive(world);
	
	// 3. Determine the "correct" state based on rules (Teacher)
	update_state_transitions(world);
	
	// 4. Train the state-selection brain (accumulates gradients)
	train_state_brain(world);
	
	// Add current state to the perception vector for the movement brain
	// The perception vector is size 22. Indices 18, 19, 20, 21 are for state.
	// We create a one-hot encoding of the current state.
	perception_vector[18] = (m_state == State::IDLE) ? 1.0f : 0.0f;
	perception_vector[19] = (m_state == State::ROAM) ? 1.0f : 0.0f;
	perception_vector[20] = (m_state == State::HUNT) ? 1.0f : 0.0f;
	perception_vector[21] = (m_state == State::FLEE) ? 1.0f : 0.0f;
	
	// 5. Train the movement brain based on the correct state (accumulates gradients)
	train_brain(world, perception_vector);
	
	// 6. Get the next action from the movement brain and update transform
	m_transform = m_brain.process_step(perception_vector);
	
	// 7. Apply gradients after accumulating for BATCH_SIZE steps
	m_gradient_accumulation_step++;
	if(m_gradient_accumulation_step >= BATCH_SIZE)
	{
		m_brain.apply_and_clear_gradients(m_learning_rate, BATCH_SIZE);
		m_state_brain.apply_and_clear_gradients(m_learning_rate, BATCH_SIZE);
		m_gradient_accumulation_step = 0;
	}
	
	// 8. Enforce world boundaries
	glm::vec3 current_pos = m_transform[3];
	float half_world_size = world.m_world_size / 2.0f;
	current_pos.x = std::clamp(current_pos.x, -half_world_size, half_world_size);
	current_pos.y = std::clamp(current_pos.y, -half_world_size, half_world_size);
	current_pos.z = std::clamp(current_pos.z, -half_world_size, half_world_size);
	m_transform[3] = glm::vec4(current_pos, 1.0f);
	m_brain.set_state(m_transform); // Keep brain state in sync after clamping
	
	// 9. Update color based on energy
	float max_energy = (m_type == AgentType::PREDATOR) ? PREDATOR_ENERGY_TO_REPRODUCE : FEEDER_ENERGY_TO_REPRODUCE;
	float energy_factor = std::clamp(m_energy / max_energy, 0.1f, 1.0f); // Clamp to avoid being invisible
	m_color = m_base_color * energy_factor;
}

// Gathers sensory information, prepares the input vector for the movement RNN,
// and updates the agent's internal world model (m_food_pos, m_threat_pos, etc.).
std::vector<float> Agent::perceive(const Simulation& world)
{
	// Reset internal world model
	m_food_pos = glm::vec3(0.0f);
	m_follow_agent_pos = glm::vec3(0.0f);
	m_threat_pos = glm::vec3(0.0f);

	// The vector now has space for new sensors and the state, which will be added in update()
	// 4 food, 4 follow, 4 threat, 1 energy, 3 boundary, 1 friendly, 1 hostile, 4 state
	std::vector<float> perception(22, 0.0f);

	glm::vec3 agent_pos = m_transform[3];
	
	// Dynamic detection radius based on world size
	const float detection_radius = world.m_world_size * 0.5f;
	const float detection_radius_sq = detection_radius * detection_radius;

	// --- Find Targets ---
	glm::vec3 closest_food_pos(0.0f);
	glm::vec3 closest_follow_pos(0.0f);
	glm::vec3 closest_threat_pos(0.0f);
	float min_food_dist_sq = std::numeric_limits<float>::max();
	float min_follow_dist_sq = ROAM_FOLLOW_RADIUS_SQ; // Roaming is short-range, keep it fixed.
	float min_threat_dist_sq = std::numeric_limits<float>::max();
	bool found_threat = false;
	bool found_follow = false;
	bool found_food = false;

	// --- Predator-specific prey tracking ---
	glm::vec3 closest_feeder_prey_pos(0.0f);
	glm::vec3 closest_predator_prey_pos(0.0f);
	float min_feeder_prey_dist_sq = std::numeric_limits<float>::max();
	float min_predator_prey_dist_sq = std::numeric_limits<float>::max();
	bool found_feeder_prey = false;
	bool found_predator_prey = false;

	// Find closest food (type-specific)
	if (m_type == AgentType::FEEDER) {
		for (const auto& food : world.m_food) {
			if (food.m_food_type != FoodType::VEGETABLE) continue;
			float dist_sq = glm::distance2(agent_pos, food.position);
			if (dist_sq < detection_radius_sq && dist_sq < min_food_dist_sq) {
				min_food_dist_sq = dist_sq;
				closest_food_pos = food.position;
				found_food = true;
			}
		}
	} else { // PREDATOR
		for (const auto& food : world.m_food) {
			if (food.m_food_type != FoodType::MEAT) continue;
			float dist_sq = glm::distance2(agent_pos, food.position);
			if (dist_sq < detection_radius_sq && dist_sq < min_food_dist_sq) {
				min_food_dist_sq = dist_sq;
				closest_food_pos = food.position;
				found_food = true;
			}
		}
	}

	// --- New Sensor: Local Agent Density ---
	float friendly_agent_count = 0.0f;
	float hostile_agent_count = 0.0f;

	// Find closest agents (threat, follow, prey) AND calculate density
	for (const auto& other : world.m_agents) {
		if (&other == this) continue;
		glm::vec3 other_pos = other.m_transform[3];
		float dist_sq = glm::distance2(agent_pos, other_pos);

		// Density Calculation
		if (dist_sq < DENSITY_RADIUS_SQ) {
			if (m_type == AgentType::FEEDER) {
				if (other.m_type == AgentType::FEEDER) friendly_agent_count++;
				else if (other.m_type == AgentType::PREDATOR) hostile_agent_count++;
			} else { // PREDATOR
				// For a predator, other predators are hostile competitors.
				if (other.m_type == AgentType::PREDATOR) hostile_agent_count++;
			}
		}

		// Closest agent logic
		if (m_type == AgentType::FEEDER) {
			// Feeder looks for predators (threat) and other feeders (follow)
			if (other.m_type == AgentType::PREDATOR && dist_sq < detection_radius_sq) {
				if (dist_sq < min_threat_dist_sq) {
					min_threat_dist_sq = dist_sq;
					closest_threat_pos = other_pos;
					found_threat = true;
				}
			} else if (other.m_type == AgentType::FEEDER && dist_sq < min_follow_dist_sq) {
				min_follow_dist_sq = dist_sq;
				closest_follow_pos = other_pos;
				found_follow = true;
			}
		} else { // PREDATOR
			// Predator looks for prey, prioritizing feeders
			if (dist_sq < detection_radius_sq) {
				if (other.m_type == AgentType::FEEDER) {
					if (dist_sq < min_feeder_prey_dist_sq) {
						min_feeder_prey_dist_sq = dist_sq;
						closest_feeder_prey_pos = other_pos;
						found_feeder_prey = true;
					}
				} else if (other.m_type == AgentType::PREDATOR) {
					if (dist_sq < min_predator_prey_dist_sq) {
						min_predator_prey_dist_sq = dist_sq;
						closest_predator_prey_pos = other_pos;
						found_predator_prey = true;
					}
				}
			}
		}
	}
	
	// --- Predator prey selection logic ---
	if (m_type == AgentType::PREDATOR) {
		if (found_feeder_prey) {
			closest_threat_pos = closest_feeder_prey_pos;
			min_threat_dist_sq = min_feeder_prey_dist_sq;
			found_threat = true;
		} else if (found_predator_prey) {
			closest_threat_pos = closest_predator_prey_pos;
			min_threat_dist_sq = min_predator_prey_dist_sq;
			found_threat = true;
		}
	}

	// --- Update Internal World Model ---
	if (found_food) m_food_pos = closest_food_pos;
	if (found_follow) m_follow_agent_pos = closest_follow_pos;
	if (found_threat) m_threat_pos = closest_threat_pos;

	// --- Fill Perception Vector for Movement Brain ---
	// Food (Direction Vector & Distance)
	if (found_food) {
		glm::vec3 direction = closest_food_pos - agent_pos;
		if (glm::length2(direction) > 1e-6f) {
			direction = glm::normalize(direction);
		} else {
			direction = glm::vec3(0.0f);
		}
		perception[0] = direction.x;
		perception[1] = direction.y;
		perception[2] = direction.z;
		perception[3] = sqrt(min_food_dist_sq);
	}

	// Follow Agent
	if (found_follow) {
		glm::vec3 direction = closest_follow_pos - agent_pos;
		if (glm::length2(direction) > 1e-6f) {
			direction = glm::normalize(direction);
		} else {
			direction = glm::vec3(0.0f);
		}
		perception[4] = direction.x;
		perception[5] = direction.y;
		perception[6] = direction.z;
		perception[7] = sqrt(min_follow_dist_sq);
	}

	// Threat/Prey Agent
	if (found_threat) {
		glm::vec3 direction = closest_threat_pos - agent_pos;
		if (glm::length2(direction) > 1e-6f) {
			direction = glm::normalize(direction);
		} else {
			direction = glm::vec3(0.0f);
		}
		perception[8] = direction.x;
		perception[9] = direction.y;
		perception[10] = direction.z;
		perception[11] = sqrt(min_threat_dist_sq);
	}

	// Own Energy
	perception[12] = m_energy / 100.0f; // Normalize energy

	// --- New Sensor: Boundary Avoidance ---
	glm::vec3 boundary_avoidance_vec(0.0f);
	float half_world_size = world.m_world_size / 2.0f;
	if (agent_pos.x > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance_vec.x -= 1.0f;
	if (agent_pos.x < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance_vec.x += 1.0f;
	if (agent_pos.y > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance_vec.y -= 1.0f;
	if (agent_pos.y < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance_vec.y += 1.0f;
	if (agent_pos.z > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance_vec.z -= 1.0f;
	if (agent_pos.z < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance_vec.z += 1.0f;
	
	if (glm::length2(boundary_avoidance_vec) > 1e-6f) {
		boundary_avoidance_vec = glm::normalize(boundary_avoidance_vec);
	}
	perception[13] = boundary_avoidance_vec.x;
	perception[14] = boundary_avoidance_vec.y;
	perception[15] = boundary_avoidance_vec.z;

	// Agent Density (normalized by a reasonable max count to keep values small)
	perception[16] = std::min(friendly_agent_count / 10.0f, 1.0f);
	perception[17] = std::min(hostile_agent_count / 10.0f, 1.0f);

	return perception;
}

// The "Teacher" - sets the correct state based on the agent's internal world model.
void Agent::update_state_transitions(const Simulation& /* world */)
{
	State next_state;
	
	if(m_type == AgentType::FEEDER)
	{
		// Priority 1: Survival. If a threat is detected, FLEE.
		if (m_threat_pos != glm::vec3(0.0f)) {
			next_state = State::FLEE;
		} 
		// Priority 2: Sustenance. If safe and hungry, HUNT for food.
		// Hunt if energy is below 90% of reproduction level, which is a better
		// indicator of need than the arbitrary INITIAL_ENERGY.
		else if (m_energy < FEEDER_ENERGY_TO_REPRODUCE * 0.9f) { 
			next_state = State::HUNT;
		} 
		// Priority 3: Social/Exploration. If safe and not hungry, ROAM.
		// Roaming is a more active and useful default state than IDLE.
		else {
			next_state = State::ROAM;
		}
	} else
	{
		// Predator Logic
		if (m_threat_pos != glm::vec3(0.0f)) {
			// Priority 1: Hunt other agents
			next_state = State::HUNT;
		} else if (m_food_pos != glm::vec3(0.0f) && m_energy < PREDATOR_ENERGY_TO_REPRODUCE * 0.8f) {
			// Priority 2: Hunt for meat if hungry and no agents are seen
			next_state = State::HUNT;
		} else {
			// Otherwise, roam to find targets
			next_state = State::ROAM;
		}
	}
	m_state = next_state;
}

// New sensor function to find the direction to the closest food of a specific type.
glm::vec3 Agent::find_closest_food_direction(const Simulation& world, size_t food_type) const
{
	glm::vec3 agent_pos = m_transform[3];
	glm::vec3 closest_food_pos(0.0f);
	float min_dist_sq = std::numeric_limits<float>::max();
	bool found_food = false;

	const float detection_radius = world.m_world_size * 0.5f;
	const float detection_radius_sq = detection_radius * detection_radius;

	for (const auto& food : world.m_food) {
		if (food.m_food_type == food_type) {
			float dist_sq = glm::distance2(agent_pos, food.position);
			if (dist_sq < detection_radius_sq && dist_sq < min_dist_sq) {
				min_dist_sq = dist_sq;
				closest_food_pos = food.position;
				found_food = true;
			}
		}
	}

	if (found_food) {
		glm::vec3 direction = closest_food_pos - agent_pos;
		if (glm::length2(direction) > 1e-6f) {
			return glm::normalize(direction);
		}
	}

	return glm::vec3(0.0f); // Return zero vector if no food is found
}

// Teacher-forcing for the movement brain using the new directional sensor approach.
void Agent::train_brain(const Simulation& world, const std::vector<float>& perception_vector)
{
	glm::vec3 desired_direction(0.0f);
	bool has_target = false;

	glm::vec3 agent_pos = m_transform[3];

	// Determine the ideal direction of movement based on the agent's state.
	switch (m_state) {
		case State::HUNT:
			if (m_type == AgentType::FEEDER) {
				// Use the sensor function to find the direction to the nearest VEGETABLE.
				desired_direction = find_closest_food_direction(world, FoodType::VEGETABLE);
			} else { // PREDATOR
				// Predators prioritize hunting other agents over scavenging.
				if (m_threat_pos != glm::vec3(0.0f)) {
					glm::vec3 dir_to_threat = m_threat_pos - agent_pos;
					if (glm::length2(dir_to_threat) > 1e-6f) {
						desired_direction = glm::normalize(dir_to_threat);
					}
				} else {
					// If no agent prey is visible, hunt for MEAT.
					desired_direction = find_closest_food_direction(world, FoodType::MEAT);
				}
			}
			has_target = (desired_direction != glm::vec3(0.0f));
			break;

		case State::FLEE:
			if (m_threat_pos != glm::vec3(0.0f)) {
				glm::vec3 dir_to_threat = m_threat_pos - agent_pos;
				glm::vec3 flee_from_threat_vec(0.0f);
				if (glm::length2(dir_to_threat) > 1e-6f) {
					flee_from_threat_vec = -glm::normalize(dir_to_threat);
				}

				// Get boundary avoidance vector from perception input
				glm::vec3 boundary_avoidance_vec(perception_vector[13], perception_vector[14], perception_vector[15]);
				
				// Combine vectors - weight boundary avoidance slightly higher to prevent getting stuck
				desired_direction = flee_from_threat_vec + (boundary_avoidance_vec * 1.5f);
				
				if (glm::length2(desired_direction) > 1e-6f) {
					desired_direction = glm::normalize(desired_direction);
					has_target = true;
				} else {
					// Fallback if vectors cancel out; just use the primary flee vector
					desired_direction = flee_from_threat_vec;
					has_target = (glm::length2(desired_direction) > 1e-6f);
				}
			}
			break;

		case State::ROAM:
			 if (m_follow_agent_pos != glm::vec3(0.0f)) {
				glm::vec3 dir_to_follow = m_follow_agent_pos - agent_pos;
				if (glm::length2(dir_to_follow) > 1e-6f) {
					desired_direction = glm::normalize(dir_to_follow);
					has_target = true;
				}
			 }
			break;

		case State::IDLE:
		default:
			// No specific target direction.
			break;
	}

	glm::vec3 agent_fwd = glm::normalize(glm::vec3(m_transform[2]));

	// --- Calculate Ideal Delta Transform based on the desired direction ---
	glm::vec3 desired_translation(0.0f);
	glm::quat desired_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity

	if (has_target) {
		// The ideal translation is to move at max speed in the desired direction.
		desired_translation = desired_direction * MAX_SPEED;
		
		// The ideal rotation is to smoothly turn to face the desired direction.
		glm::quat target_rot = glm::rotation(agent_fwd, desired_direction);
		desired_rotation = glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), target_rot, ROTATION_LERP_FACTOR);
	} else {
		// For IDLE or ROAM without a target, just move forward slowly.
		desired_translation = agent_fwd * (MAX_SPEED * 0.2f);
	}

	// --- Create Target Parameter Vector for the RNN ---
	std::vector<float> target_params(7);
	target_params[0] = desired_translation.x;
	target_params[1] = desired_translation.y;
	target_params[2] = desired_translation.z;
	target_params[3] = desired_rotation.x;
	target_params[4] = desired_rotation.y;
	target_params[5] = desired_rotation.z;
	target_params[6] = desired_rotation.w;

	// --- Train the Brain ---
	m_brain.train_step(perception_vector, target_params, ROTATION_WEIGHT);
}

// New function to train the state-selection RNN
void Agent::train_state_brain(const Simulation& /* world */)
{
	// 1. Create input vector for the state brain.
	std::vector<double> state_brain_input_vec;
	state_brain_input_vec.reserve(STATE_BRAIN_INPUT_DIM);

	// --- Part A: Energy Input (One-hot encoded bin) ---
	std::vector<double> energy_input_vec(NUM_ENERGY_BINS, 0.0);
	float max_energy = (m_type == AgentType::PREDATOR) ? PREDATOR_ENERGY_TO_REPRODUCE : FEEDER_ENERGY_TO_REPRODUCE;
	float energy_ratio = std::clamp(m_energy / max_energy, 0.0f, 1.0f);
	// Ensure the bin index is within bounds [0, NUM_ENERGY_BINS - 1]
	int energy_bin_index = static_cast<int>(energy_ratio * (NUM_ENERGY_BINS - 0.0001f));
	energy_input_vec[energy_bin_index] = 1.0;
	state_brain_input_vec.insert(state_brain_input_vec.end(), energy_input_vec.begin(), energy_input_vec.end());

	// --- Part B: Distance to Enemy Input (One-hot encoded bin) ---
	std::vector<double> distance_input_vec(NUM_DISTANCE_BINS, 0.0);
	float distance_to_threat = std::numeric_limits<float>::max();

	if (m_threat_pos != glm::vec3(0.0f)) {
		distance_to_threat = glm::distance(glm::vec3(m_transform[3]), m_threat_pos);
	}

	float max_detection_radius = (m_type == AgentType::FEEDER) ? 
								 sqrt(PREDATOR_DETECTION_RADIUS_SQ) : 
								 sqrt(PREDATOR_HUNT_RADIUS_SQ);
	
	float distance_ratio = std::clamp(distance_to_threat / max_detection_radius, 0.0f, 1.0f);
	int distance_bin_index = static_cast<int>(distance_ratio * (NUM_DISTANCE_BINS - 0.0001f));
	distance_input_vec[distance_bin_index] = 1.0;
	state_brain_input_vec.insert(state_brain_input_vec.end(), distance_input_vec.begin(), distance_input_vec.end());

	// --- Part C: Distance to Food Input (One-hot encoded bin) ---
	std::vector<double> food_distance_input_vec(NUM_FOOD_DISTANCE_BINS, 0.0);
	float distance_to_food = std::numeric_limits<float>::max();

	if (m_food_pos != glm::vec3(0.0f)) {
		distance_to_food = glm::distance(glm::vec3(m_transform[3]), m_food_pos);
	}
	
	// Use a general detection radius for food. PREDATOR_DETECTION_RADIUS_SQ is a good general "far" distance.
	float max_food_detection_radius = sqrt(FEEDER_FOOD_DETECTION_RADIUS_SQ);
	float food_distance_ratio = std::clamp(distance_to_food / max_food_detection_radius, 0.0f, 1.0f);
	int food_distance_bin_index = static_cast<int>(food_distance_ratio * (NUM_FOOD_DISTANCE_BINS - 0.0001f));
	food_distance_input_vec[food_distance_bin_index] = 1.0;
	state_brain_input_vec.insert(state_brain_input_vec.end(), food_distance_input_vec.begin(), food_distance_input_vec.end());

	// 2. Get the one-hot target vector for the current "correct" state.
	size_t target_state_id = static_cast<size_t>(m_state);
	const auto& target_state_vec = state_tokenizer.find_vector(target_state_id);

	// 3. Perform a training step.
	m_state_brain.reset_state();
	std::vector<std::vector<double>> inputs_sequence = { state_brain_input_vec };
	std::vector<double> predicted_output = m_state_brain.forward_pass(state_brain_input_vec);
	std::vector<std::vector<double>> outputs_sequence = { predicted_output };
	std::vector<std::vector<double>> expected_sequence = { target_state_vec };
	m_state_brain.backpropagate(inputs_sequence, outputs_sequence, expected_sequence);
}

float Agent::get_learning_rate() const
{
	return m_learning_rate;
}

size_t Agent::get_state_brain_hidden_dim() const
{
	return m_state_brain_hidden_dim;
}

bool Agent::isDead() const
{
	return m_energy <= 0.0f || m_age > MAX_AGE;
}

bool Agent::canReproduce() const {
	if (m_type == AgentType::PREDATOR) {
		return m_energy >= PREDATOR_ENERGY_TO_REPRODUCE;
	} else {
		return m_energy >= FEEDER_ENERGY_TO_REPRODUCE;
	}
}

Agent Agent::reproduce() {
	// Halve parent's energy for the child
	m_energy /= 2.0f;
	float child_energy = m_energy;

	// Create a copy of the parent's brain
	ls3DRNN child_brain = m_brain;

	// Mutate the child's brain
	child_brain.mutate(MUTATION_RATE, MUTATION_AMOUNT);

	// Child spawns at parent's location
	glm::mat4 child_transform = m_transform;

	// Create the new agent. Learning rate is randomized by passing -1.
	Agent child(std::move(child_brain), child_transform, m_type, -1.0f, child_energy);
	
	// The child's state brain is newly initialized, it will learn on its own.
	return child;
}
