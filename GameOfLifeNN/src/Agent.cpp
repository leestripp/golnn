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
// Distribution for spawning offset to prevent clumping
static std::uniform_real_distribution<float> offset_dist(-1.0f, 1.0f);

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
	// Sized here rather than in the init list: m_perception is declared before
	// m_state_brain in the header, so member-init order would warn.
	m_perception.assign(PERCEPTION_VECTOR_SIZE, 0.0f);
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
	std::vector<float>& perception_vector = perceive(world);
	
	// 3. Determine the "correct" state based on rules (Teacher)
	update_state_transitions(world);
	
	// 4. Train the state-selection brain (accumulates gradients)
	train_state_brain(world);
	
	// Add current state to the perception vector for the movement brain
	// The perception vector is size 31. Indices 27, 28, 29, 30 are for state.
	perception_vector[27] = (m_state == State::IDLE) ? 1.0f : 0.0f;
	perception_vector[28] = (m_state == State::ROAM) ? 1.0f : 0.0f;
	perception_vector[29] = (m_state == State::HUNT) ? 1.0f : 0.0f;
	perception_vector[30] = (m_state == State::FLEE) ? 1.0f : 0.0f;
	
	// 5. Train the movement brain based on the correct state (accumulates gradients)
	train_brain(perception_vector);
	
	// 6. Get the next action from the movement brain and update transform
	m_transform = m_brain.process_step(perception_vector, MAX_SPEED);
	
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

// New constants for sensors
constexpr float MAX_SOUND_RADIUS_SQ = 900.0f; // 30 units
constexpr float MAX_SCENT_RADIUS_SQ = 900.0f; // 30 units

// Gathers sensory information, prepares the input vector for the movement RNN,
// and updates the agent's internal world model (m_food_pos, m_threat_pos, etc.).
std::vector<float>& Agent::perceive(const Simulation& world)
{
	// Reset internal world model
	m_food_pos = glm::vec3(0.0f);
	m_follow_agent_pos = glm::vec3(0.0f);
	m_threat_pos = glm::vec3(0.0f);

	// The vector now has space for all new sensors. The state will be added in update().
	// See Agent.h for PERCEPTION_VECTOR_SIZE definition (31)
	std::vector<float>& perception = m_perception;
	std::fill(perception.begin(), perception.end(), 0.0f);

	glm::vec3 agent_pos = m_transform[3];
	
	// Dynamic detection radius for VISUAL sensors
	const float detection_radius = world.m_world_size * 0.5f;
	const float detection_radius_sq = detection_radius * detection_radius;

	// --- SENSOR: Visual Targets ---
	glm::vec3 closest_food_pos(0.0f);
	glm::vec3 closest_follow_pos(0.0f);
	glm::vec3 closest_threat_pos(0.0f);
	float min_food_dist_sq = std::numeric_limits<float>::max();
	float min_follow_dist_sq = ROAM_FOLLOW_RADIUS_SQ;
	float min_threat_dist_sq = std::numeric_limits<float>::max();
	bool found_threat = false;
	bool found_follow = false;
	bool found_food = false;

	// Predator-specific prey tracking
	glm::vec3 closest_feeder_prey_pos(0.0f);
	glm::vec3 closest_predator_prey_pos(0.0f);
	float min_feeder_prey_dist_sq = std::numeric_limits<float>::max();
	float min_predator_prey_dist_sq = std::numeric_limits<float>::max();
	bool found_feeder_prey = false;
	bool found_predator_prey = false;

	// Find closest food via the target type's own spatial grid: feeders track
	// vegetables, predators track meat, and each grid holds only that type.
	const SpatialGrid& target_food_grid =
		(m_type == AgentType::FEEDER) ? world.m_vegetable_grid : world.m_meat_grid;
	target_food_grid.visit(agent_pos, detection_radius, [&](size_t i) {
		const Food& food = world.m_food[i];
		float dist_sq = glm::distance2(agent_pos, food.position);
		if (dist_sq < detection_radius_sq && dist_sq < min_food_dist_sq) {
			min_food_dist_sq = dist_sq;
			closest_food_pos = food.position;
			found_food = true;
		}
	});

	// --- SENSOR: Local Agent Density & Visual Agent Tracking ---
	// Each sensor visits only the grid holding the types it can detect, so a
	// feeder's threat scan touches the (small) predator grid instead of every
	// agent in the world.
	float friendly_agent_count = 0.0f;
	float hostile_agent_count = 0.0f;

	if (m_type == AgentType::FEEDER) {
		// Threats: closest predator within visual range (also counts local
		// hostile density — same visit, extra comparison).
		world.m_predator_grid.visit(agent_pos, detection_radius, [&](size_t i) {
			const Agent& other = world.m_agents[i];
			glm::vec3 other_pos = other.m_transform[3];
			float dist_sq = glm::distance2(agent_pos, other_pos);
			if (dist_sq < DENSITY_RADIUS_SQ) hostile_agent_count++;
			if (dist_sq < detection_radius_sq && dist_sq < min_threat_dist_sq) {
				min_threat_dist_sq = dist_sq;
				closest_threat_pos = other_pos;
				found_threat = true;
			}
		});
		// Friends: closest feeder within follow range (ROAM_FOLLOW_RADIUS_SQ).
		world.m_feeder_grid.visit(agent_pos, sqrt(ROAM_FOLLOW_RADIUS_SQ), [&](size_t i) {
			const Agent& other = world.m_agents[i];
			if (&other == this) return;
			glm::vec3 other_pos = other.m_transform[3];
			float dist_sq = glm::distance2(agent_pos, other_pos);
			if (dist_sq < DENSITY_RADIUS_SQ) friendly_agent_count++;
			if (dist_sq < min_follow_dist_sq) {
				min_follow_dist_sq = dist_sq;
				closest_follow_pos = other_pos;
				found_follow = true;
			}
		});
	} else { // PREDATOR
		// Prey: closest feeder within visual range.
		world.m_feeder_grid.visit(agent_pos, detection_radius, [&](size_t i) {
			const Agent& other = world.m_agents[i];
			glm::vec3 other_pos = other.m_transform[3];
			float dist_sq = glm::distance2(agent_pos, other_pos);
			if (dist_sq < detection_radius_sq && dist_sq < min_feeder_prey_dist_sq) {
				min_feeder_prey_dist_sq = dist_sq;
				closest_feeder_prey_pos = other_pos;
				found_feeder_prey = true;
			}
		});
		// Rivals: closest other predator within visual range (also hostile density).
		world.m_predator_grid.visit(agent_pos, detection_radius, [&](size_t i) {
			const Agent& other = world.m_agents[i];
			if (&other == this) return;
			glm::vec3 other_pos = other.m_transform[3];
			float dist_sq = glm::distance2(agent_pos, other_pos);
			if (dist_sq < DENSITY_RADIUS_SQ) hostile_agent_count++;
			if (dist_sq < detection_radius_sq && dist_sq < min_predator_prey_dist_sq) {
				min_predator_prey_dist_sq = dist_sq;
				closest_predator_prey_pos = other_pos;
				found_predator_prey = true;
			}
		});
	}
	
	// Predator prey selection logic
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

	// --- Fill Perception Vector ---
	// [0-3] Closest Food (Visual)
	if (found_food) {
		glm::vec3 direction = closest_food_pos - agent_pos;
		if (glm::length2(direction) > 1e-6f) direction = glm::normalize(direction);
		else direction = glm::vec3(0.0f);
		perception[0] = direction.x;
		perception[1] = direction.y;
		perception[2] = direction.z;
		perception[3] = sqrt(min_food_dist_sq);
	}
	// [4-7] Closest Threat/Prey (Visual)
	if (found_threat) {
		glm::vec3 direction = closest_threat_pos - agent_pos;
		if (glm::length2(direction) > 1e-6f) direction = glm::normalize(direction);
		else direction = glm::vec3(0.0f);
		perception[4] = direction.x;
		perception[5] = direction.y;
		perception[6] = direction.z;
		perception[7] = sqrt(min_threat_dist_sq);
	}
	// [8-11] Closest Friend (Visual)
	if (found_follow) {
		glm::vec3 direction = closest_follow_pos - agent_pos;
		if (glm::length2(direction) > 1e-6f) direction = glm::normalize(direction);
		else direction = glm::vec3(0.0f);
		perception[8] = direction.x;
		perception[9] = direction.y;
		perception[10] = direction.z;
		perception[11] = sqrt(min_follow_dist_sq);
	}

	// --- SENSOR: Scent Gradient ---
	// [12-15] Strongest Scent (dir xyz, intensity)
	glm::vec3 strongest_scent_dir(0.0f);
	float max_scent_intensity = 0.0f;
	
	// Scent radius (30) matches the spatial grid cell size, so this touches
	// at most one ring of neighbouring cells.
	const float scent_radius = sqrt(MAX_SCENT_RADIUS_SQ);
	if (m_type == AgentType::FEEDER) {
		world.m_vegetable_grid.visit(agent_pos, scent_radius, [&](size_t i) {
			const Food& food = world.m_food[i];
			float dist_sq = glm::distance2(agent_pos, food.position);
			if (dist_sq < MAX_SCENT_RADIUS_SQ) {
				float intensity = 1.0f / (dist_sq + 1.0f);
				if (intensity > max_scent_intensity) {
					max_scent_intensity = intensity;
					strongest_scent_dir = food.position - agent_pos;
				}
			}
		});
	} else { // PREDATOR
		world.m_meat_grid.visit(agent_pos, scent_radius, [&](size_t i) { // Smell MEAT
			const Food& food = world.m_food[i];
			float dist_sq = glm::distance2(agent_pos, food.position);
			if (dist_sq < MAX_SCENT_RADIUS_SQ) {
				float intensity = 1.0f / (dist_sq + 1.0f);
				if (intensity > max_scent_intensity) {
					max_scent_intensity = intensity;
					strongest_scent_dir = food.position - agent_pos;
				}
			}
		});
		world.m_feeder_grid.visit(agent_pos, scent_radius, [&](size_t i) { // Smell FEEDERS
			const Agent& other = world.m_agents[i];
			if (&other == this) return;
			float dist_sq = glm::distance2(agent_pos, glm::vec3(other.m_transform[3]));
			if (dist_sq < MAX_SCENT_RADIUS_SQ) {
				float intensity = 1.5f / (dist_sq + 1.0f); // Agents are a stronger scent
				if (intensity > max_scent_intensity) {
					max_scent_intensity = intensity;
					strongest_scent_dir = glm::vec3(other.m_transform[3]) - agent_pos;
				}
			}
		});
	}

	if (max_scent_intensity > 0.0f) {
		if (glm::length2(strongest_scent_dir) > 1e-6f) strongest_scent_dir = glm::normalize(strongest_scent_dir);
		perception[12] = strongest_scent_dir.x;
		perception[13] = strongest_scent_dir.y;
		perception[14] = strongest_scent_dir.z;
		perception[15] = std::min(max_scent_intensity, 1.0f);
	}

	// --- SENSOR: Auditory ---
	// [16-19] Loudest Sound (dir xyz, intensity)
	glm::vec3 loudest_sound_dir(0.0f);
	float max_sound_intensity = 0.0f;

	for (const auto& sound : world.m_sound_events) {
		float dist_sq = glm::distance2(agent_pos, sound.position);
		if (dist_sq < MAX_SOUND_RADIUS_SQ) {
			float perceived_intensity = sound.intensity / (dist_sq + 1.0f);
			if (perceived_intensity > max_sound_intensity) {
				max_sound_intensity = perceived_intensity;
				loudest_sound_dir = sound.position - agent_pos;
			}
		}
	}

	if (max_sound_intensity > 0.0f) {
		if (glm::length2(loudest_sound_dir) > 1e-6f) loudest_sound_dir = glm::normalize(loudest_sound_dir);
		perception[16] = loudest_sound_dir.x;
		perception[17] = loudest_sound_dir.y;
		perception[18] = loudest_sound_dir.z;
		perception[19] = std::min(max_sound_intensity, 1.0f);
	}
	
	// --- SENSOR: Boundary Avoidance ---
	// [20-22] Boundary Avoidance (dir xyz)
	glm::vec3 boundary_avoidance_vec(0.0f);
	float half_world_size = world.m_world_size / 2.0f;
	if (agent_pos.x > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance_vec.x -= 1.0f;
	if (agent_pos.x < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance_vec.x += 1.0f;
	if (agent_pos.y > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance_vec.y -= 1.0f;
	if (agent_pos.y < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance_vec.y += 1.0f;
	if (agent_pos.z > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance_vec.z -= 1.0f;
	if (agent_pos.z < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance_vec.z += 1.0f;
	
	if (glm::length2(boundary_avoidance_vec) > 1e-6f) boundary_avoidance_vec = glm::normalize(boundary_avoidance_vec);
	perception[20] = boundary_avoidance_vec.x;
	perception[21] = boundary_avoidance_vec.y;
	perception[22] = boundary_avoidance_vec.z;

	// --- SENSOR: Local Agent Density ---
	// [23] Friendly Agent Density
	// [24] Hostile Agent Density
	perception[23] = std::min(friendly_agent_count / 10.0f, 1.0f);
	perception[24] = std::min(hostile_agent_count / 10.0f, 1.0f);

	// --- SENSOR: Internal State ---
	// [25] Energy
	float max_energy = (m_type == AgentType::PREDATOR) ? PREDATOR_ENERGY_TO_REPRODUCE : FEEDER_ENERGY_TO_REPRODUCE;
    perception[25] = std::clamp(m_energy / max_energy, 0.0f, 1.0f);
    
    // [26] Age
    perception[26] = static_cast<float>(m_age) / static_cast<float>(MAX_AGE);
    
    // [27-30] are for state, added in update()

	return perception;
}

// The "Teacher" - sets the correct state based on the agent's internal world model.
void Agent::update_state_transitions(const Simulation& /* world */)
{
	State next_state;
	
	if(m_type == AgentType::FEEDER)
	{
		if(m_threat_pos != glm::vec3(0.0f))
		{
			// Priority 1: Survival. If a threat is detected, FLEE.
			next_state = State::FLEE;
		} else if(m_energy < FEEDER_ENERGY_TO_REPRODUCE * 0.9f && m_food_pos != glm::vec3(0.0f))
		{
			// Priority 2: Sustenance. If safe and hungry, HUNT for food.
			// Hunt if energy is below 90% of reproduction level, which is a better
			// indicator of need than the arbitrary INITIAL_ENERGY.
			next_state = State::HUNT;
		} else
		{
			// Priority 3: Social/Exploration. If safe and not hungry, ROAM.
			// Roaming is a more active and useful default state than IDLE.
			next_state = State::ROAM;
		}
	} else
	{
		// Predator Logic
		if(m_threat_pos != glm::vec3(0.0f))
		{
			// Priority 1: Hunt other Feeder agents
			next_state = State::HUNT;
		} else if(m_food_pos != glm::vec3(0.0f) && m_energy < PREDATOR_ENERGY_TO_REPRODUCE)
		{
			// Priority 2: Hunt for meat if hungry and no agents are seen
			next_state = State::HUNT;
		} else
		{
			// Otherwise, roam to find targets
			next_state = State::ROAM;
		}
	}
	m_state = next_state;
}

// Teacher-forcing for the movement brain using the new directional sensor approach.
void Agent::train_brain(const std::vector<float>& perception_vector)
{
	glm::vec3 primary_direction(0.0f);
	bool has_target = false;

	glm::vec3 agent_pos = m_transform[3];

	// Step 1: Determine the ideal PRIMARY direction based on state (the main goal).
	switch (m_state) {
		case State::HUNT:
			// m_food_pos already holds the closest target-type food (vegetable
			// for feeders, meat for predators) found by perceive() this tick —
			// reuse it instead of re-scanning the food grid.
			if (m_type == AgentType::FEEDER || m_threat_pos == glm::vec3(0.0f)) {
				if (m_food_pos != glm::vec3(0.0f)) {
					glm::vec3 dir_to_food = m_food_pos - agent_pos;
					if (glm::length2(dir_to_food) > 1e-6f) {
						primary_direction = glm::normalize(dir_to_food);
					}
				}
			} else { // PREDATOR chasing prey
				glm::vec3 dir_to_threat = m_threat_pos - agent_pos;
				if (glm::length2(dir_to_threat) > 1e-6f) {
					primary_direction = glm::normalize(dir_to_threat);
				}
			}
			has_target = (primary_direction != glm::vec3(0.0f));
			break;

		case State::FLEE:
			if (m_threat_pos != glm::vec3(0.0f)) {
				// Flee directly away from the threat.
				glm::vec3 dir_from_threat = agent_pos - m_threat_pos;
				if (glm::length2(dir_from_threat) > 1e-6f) {
					primary_direction = glm::normalize(dir_from_threat);
				}
			}
			has_target = (primary_direction != glm::vec3(0.0f));
			break;

		case State::ROAM:
			 if (m_follow_agent_pos != glm::vec3(0.0f)) {
				glm::vec3 dir_to_follow = m_follow_agent_pos - agent_pos;
				if (glm::length2(dir_to_follow) > 1e-6f) {
					primary_direction = glm::normalize(dir_to_follow);
					has_target = true;
				}
			 }
			 // If no target, primary_direction remains (0,0,0).
			break;

		case State::IDLE:
		default:
			// No primary direction.
			break;
	}

	// Step 2: Get the boundary avoidance vector from sensors.
	glm::vec3 boundary_avoidance_vec(perception_vector[20], perception_vector[21], perception_vector[22]);
	glm::vec3 desired_direction = primary_direction;

	// Step 3: Combine the primary goal with boundary avoidance.
	// This ensures the agent learns to curve away from walls instead of running into them.
	if (glm::length2(boundary_avoidance_vec) > 1e-6f) {
		// The weight makes avoidance a stronger influence than the primary goal when near a wall.
		desired_direction += boundary_avoidance_vec * 1.5f;
	}

	// For ROAM without a follow target, the main goal IS to wander away from walls.
	if (m_state == State::ROAM && !has_target) {
		desired_direction = boundary_avoidance_vec;
	}

	// Step 4: Normalize the final direction vector.
	if (glm::length2(desired_direction) > 1e-6f) {
		desired_direction = glm::normalize(desired_direction);
		has_target = true; // We now have a valid direction.
	} else if (has_target) {
		// Fallback in case the vectors perfectly cancel out; just use the primary goal.
		desired_direction = primary_direction;
		has_target = (glm::length2(desired_direction) > 1e-6f);
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
void Agent::train_state_brain(const Simulation& world)
{
	// 1. Create the input vector for the state brain directly in the reused
	//    scratch buffer (three one-hot bin blocks, no per-tick allocations).
	std::vector<float>& input = m_state_input_scratch;
	input.assign(STATE_BRAIN_INPUT_DIM, 0.0f);

	// Use the dynamic detection radius for the binning
	const float max_detection_radius = world.m_world_size * 0.5f;

	// --- Part A: Energy Input (One-hot encoded bin) ---
	const float max_energy = (m_type == AgentType::PREDATOR) ? PREDATOR_ENERGY_TO_REPRODUCE : FEEDER_ENERGY_TO_REPRODUCE;
	const float energy_ratio = std::clamp(m_energy / max_energy, 0.0f, 1.0f);
	const int energy_bin_index = static_cast<int>(energy_ratio * (NUM_ENERGY_BINS - 0.0001f));
	input[energy_bin_index] = 1.0f;

	// --- Part B: Distance to Enemy Input (One-hot encoded bin) ---
	float distance_to_threat = std::numeric_limits<float>::max();
	if (m_threat_pos != glm::vec3(0.0f)) {
		distance_to_threat = glm::distance(glm::vec3(m_transform[3]), m_threat_pos);
	}
	const float distance_ratio = std::clamp(distance_to_threat / max_detection_radius, 0.0f, 1.0f);
	const int distance_bin_index = NUM_ENERGY_BINS + static_cast<int>(distance_ratio * (NUM_DISTANCE_BINS - 0.0001f));
	input[distance_bin_index] = 1.0f;

	// --- Part C: Distance to Food Input (One-hot encoded bin) ---
	float distance_to_food = std::numeric_limits<float>::max();
	if (m_food_pos != glm::vec3(0.0f)) {
		distance_to_food = glm::distance(glm::vec3(m_transform[3]), m_food_pos);
	}
	const float food_distance_ratio = std::clamp(distance_to_food / max_detection_radius, 0.0f, 1.0f);
	const int food_bin_index = NUM_ENERGY_BINS + NUM_DISTANCE_BINS
		+ static_cast<int>(food_distance_ratio * (NUM_FOOD_DISTANCE_BINS - 0.0001f));
	input[food_bin_index] = 1.0f;

	// 2. One-hot target vector for the current "correct" state (teacher).
	const size_t target_state_id = static_cast<size_t>(m_state);
	const auto& target_state_vec = state_tokenizer.find_vector(target_state_id);

	// 3. Single-step training (resets state, forward, length-1 BPTT).
	m_state_brain.train_step(input, target_state_vec);
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

	// Child spawns near parent's location with a small random offset to prevent clumping
	glm::mat4 child_transform = m_transform;
    glm::vec3 offset(offset_dist(agent_gen), offset_dist(agent_gen), offset_dist(agent_gen));
    child_transform = glm::translate(child_transform, offset * 2.0f); // Offset by up to 2 units

	// Create the new agent. Learning rate is randomized by passing -1.
	// The child's state brain is also randomized.
	Agent child(std::move(child_brain), child_transform, m_type, -1.0f, child_energy);
	
	// Mutate the state brain as well
	// NOTE: lsRNN does not have a mutate function, so the child gets a new random one.
	// This is a valid evolutionary strategy.
	
	return child;
}
