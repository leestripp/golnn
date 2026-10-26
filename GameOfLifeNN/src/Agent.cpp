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

namespace
{
	size_t random_state_neuron_count(std::mt19937& rng)
	{
		std::uniform_int_distribution<size_t> dist(Agent::MIN_STATE_NEURONS, Agent::MAX_STATE_NEURONS);
		return dist(rng);
	}

	const lsVecTokenizer& state_tokenizer()
	{
		// Function-local static initialization is thread-safe and has no shared
		// mutable initialization flag.
		static const lsVecTokenizer tokenizer = [] {
			lsVecTokenizer result(Agent::NUM_STATES);
			result.add_token_onehot(static_cast<size_t>(Agent::State::IDLE));
			result.add_token_onehot(static_cast<size_t>(Agent::State::ROAM));
			result.add_token_onehot(static_cast<size_t>(Agent::State::HUNT));
			result.add_token_onehot(static_cast<size_t>(Agent::State::FLEE));
			return result;
		}();
		return tokenizer;
	}
}


// Agent constructor
Agent::Agent(ls3DRNN brain, const glm::mat4& initial_transform, AgentType type, float learning_rate, float initial_energy, size_t state_hidden_dim, std::uint32_t random_seed, BrainTier brain_tier) :
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
	m_brain_tier(brain_tier),
	m_energy_decay_rate(FEEDER_ENERGY_DECAY),
	m_rng(random_seed),
	m_state_brain_hidden_dim(state_hidden_dim == 0 ? random_state_neuron_count(m_rng) : state_hidden_dim),
	m_state_brain(STATE_BRAIN_INPUT_DIM, m_state_brain_hidden_dim, NUM_STATES, Task::CLASSIFICATION, m_rng()),
	m_gradient_accumulation_step(0)
{
	// Sized here rather than in the init list: m_perception is declared before
	// m_state_brain in the header, so member-init order would warn.
	m_perception.assign(PERCEPTION_VECTOR_SIZE, 0.0f);
	m_movement_target.assign(7, 0.0f);
	// Assign learning rate: random if not provided, otherwise use the given value.
	if(learning_rate < 0.0f)
	{
		std::uniform_real_distribution<float> lr_dist(0.0001f, 0.1f);
		m_learning_rate = lr_dist(m_rng);
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
	m_previous_tick_energy = m_energy;
	
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

// Convenience scalar update used by tests and custom brains. Simulation splits
// these phases so standardized movement brains can be trained as a batch.
void Agent::update(const Simulation& world)
{
	if(!prepare_update(world)) return;
	train_movement();
	finish_update(world);
}

bool Agent::prepare_update(const Simulation& world)
{
	m_movement_pending = false;
	m_velocity = (glm::vec3(m_transform[3]) - glm::vec3(m_prev_transform[3])) / MAX_SPEED;
	m_age++;
	m_energy -= m_energy_decay_rate;
	m_energy_trend = 0.95f * m_energy_trend + 0.05f * (m_energy - m_previous_tick_energy);
	m_previous_tick_energy = m_energy;
	if(isDead()) return false;

	m_prev_transform = m_transform;
	std::vector<float>& perception_vector = perceive(world);
	update_state_transitions(world);

	std::fill(perception_vector.begin() + perception::STATE_ONE_HOT,
		perception_vector.begin() + perception::STATE_ONE_HOT + NUM_STATES, 0.0f);
	perception_vector[perception::STATE_ONE_HOT + static_cast<std::size_t>(m_state)] = 1.0f;
	prepare_brain_target(perception_vector);
	train_state_brain(world);
	m_movement_pending = true;
	return true;
}

void Agent::train_movement()
{
	if(m_movement_pending)
		m_brain.train_step(m_perception, m_movement_target, ROTATION_WEIGHT);
}

void Agent::finish_update(const Simulation& world)
{
	if(!m_movement_pending) return;
	m_transform = m_brain.process_step(m_perception, MAX_SPEED);

	m_gradient_accumulation_step++;
	if(m_gradient_accumulation_step >= BATCH_SIZE)
	{
		m_brain.apply_and_clear_gradients(m_learning_rate, BATCH_SIZE);
		m_state_brain.apply_and_clear_gradients(m_learning_rate, BATCH_SIZE);
		m_gradient_accumulation_step = 0;
	}

	glm::vec3 current_pos = m_transform[3];
	const float half_world_size = world.m_world_size / 2.0f;
	current_pos.x = std::clamp(current_pos.x, -half_world_size, half_world_size);
	current_pos.y = std::clamp(current_pos.y, -half_world_size, half_world_size);
	current_pos.z = std::clamp(current_pos.z, -half_world_size, half_world_size);
	m_transform[3] = glm::vec4(current_pos, 1.0f);
	m_brain.set_state(m_transform);

	const float max_energy = (m_type == AgentType::PREDATOR) ? PREDATOR_ENERGY_TO_REPRODUCE : FEEDER_ENERGY_TO_REPRODUCE;
	const float energy_factor = std::clamp(m_energy / max_energy, 0.1f, 1.0f);
	m_color = m_base_color * energy_factor;
	m_movement_pending = false;
}

// New constants for sensors
constexpr float MAX_SOUND_RADIUS_SQ = 900.0f; // 30 units
constexpr float MAX_SCENT_RADIUS_SQ = 900.0f; // 30 units

// Gathers sensory information in a small number of spatial-grid visits. Vision
// covers a wide 300-degree field; scent and hearing remain omnidirectional.
std::vector<float>& Agent::perceive(const Simulation& world)
{
	m_food_pos = glm::vec3(0.0f);
	m_follow_agent_pos = glm::vec3(0.0f);
	m_threat_pos = glm::vec3(0.0f);
	auto& values = m_perception;
	std::fill(values.begin(), values.end(), 0.0f);

	const glm::vec3 agent_pos = m_transform[3];
	const glm::vec3 agent_fwd = glm::normalize(glm::vec3(m_transform[2]));
	const float detection_radius = world.m_world_size * 0.5f;
	const float detection_radius_sq = detection_radius * detection_radius;
	constexpr float fov_cosine_squared = 0.75f; // rear 60 degrees are blind
	const auto is_visible = [&](const glm::vec3& offset, const float distance_sq) {
		if(distance_sq < 1.0f) return true; // close peripheral awareness
		const float forward_dot = glm::dot(agent_fwd, offset);
		if(forward_dot >= 0.0f) return true;
		return forward_dot * forward_dot <= fov_cosine_squared * distance_sq;
	};

	glm::vec3 closest_food_pos(0.0f);
	glm::vec3 closest_follow_pos(0.0f);
	glm::vec3 closest_threat_pos(0.0f);
	float min_food_dist_sq = std::numeric_limits<float>::max();
	float min_follow_dist_sq = ROAM_FOLLOW_RADIUS_SQ;
	float min_threat_dist_sq = std::numeric_limits<float>::max();
	bool found_threat = false;
	bool found_follow = false;
	bool found_food = false;
	float friendly_agent_count = 0.0f;
	float hostile_agent_count = 0.0f;
	float food_density_count = 0.0f;
	glm::vec3 strongest_scent_dir(0.0f);
	float max_scent_intensity = 0.0f;

	// One food visit gathers the nearest visible target, local density, and the
	// strongest scent instead of scanning the same cells again for each sense.
	const SpatialGrid& target_food_grid =
		(m_type == AgentType::FEEDER) ? world.m_vegetable_grid : world.m_meat_grid;
	target_food_grid.visit(agent_pos, detection_radius, [&](const std::size_t i) {
		const Food& food = world.m_food[i];
		const glm::vec3 offset = food.position - agent_pos;
		const float dist_sq = glm::length2(offset);
		if(dist_sq < FOOD_DENSITY_RADIUS_SQ) ++food_density_count;
		if(dist_sq < MAX_SCENT_RADIUS_SQ)
		{
			const float intensity = 1.0f / (dist_sq + 1.0f);
			if(intensity > max_scent_intensity)
			{
				max_scent_intensity = intensity;
				strongest_scent_dir = offset;
			}
		}
		if(dist_sq < detection_radius_sq && dist_sq < min_food_dist_sq && is_visible(offset, dist_sq))
		{
			min_food_dist_sq = dist_sq;
			closest_food_pos = food.position;
			found_food = true;
		}
	});

	if(m_type == AgentType::FEEDER)
	{
		world.m_predator_grid.visit(agent_pos, detection_radius, [&](const std::size_t i) {
			const glm::vec3 other_pos = world.sensed_agent_transform(i)[3];
			const glm::vec3 offset = other_pos - agent_pos;
			const float dist_sq = glm::length2(offset);
			if(dist_sq < DENSITY_RADIUS_SQ) ++hostile_agent_count;
			if(dist_sq < detection_radius_sq && dist_sq < min_threat_dist_sq && is_visible(offset, dist_sq))
			{
				min_threat_dist_sq = dist_sq;
				closest_threat_pos = other_pos;
				found_threat = true;
			}
		});
		world.m_feeder_grid.visit(agent_pos, std::sqrt(ROAM_FOLLOW_RADIUS_SQ), [&](const std::size_t i) {
			const Agent& other = world.m_agents[i];
			if(&other == this) return;
			const glm::vec3 other_pos = world.sensed_agent_transform(i)[3];
			const glm::vec3 offset = other_pos - agent_pos;
			const float dist_sq = glm::length2(offset);
			if(dist_sq < DENSITY_RADIUS_SQ) ++friendly_agent_count;
			if(dist_sq < min_follow_dist_sq && is_visible(offset, dist_sq))
			{
				min_follow_dist_sq = dist_sq;
				closest_follow_pos = other_pos;
				found_follow = true;
			}
		});
	}
	else
	{
		glm::vec3 closest_feeder_prey_pos(0.0f);
		glm::vec3 closest_predator_prey_pos(0.0f);
		float min_feeder_prey_dist_sq = std::numeric_limits<float>::max();
		float min_predator_prey_dist_sq = std::numeric_limits<float>::max();
		bool found_feeder_prey = false;
		bool found_predator_prey = false;

		// Prey vision and prey scent share one feeder-grid visit.
		world.m_feeder_grid.visit(agent_pos, detection_radius, [&](const std::size_t i) {
			const Agent& other = world.m_agents[i];
			if(&other == this) return;
			const glm::vec3 other_pos = world.sensed_agent_transform(i)[3];
			const glm::vec3 offset = other_pos - agent_pos;
			const float dist_sq = glm::length2(offset);
			if(dist_sq < MAX_SCENT_RADIUS_SQ)
			{
				const float intensity = 1.5f / (dist_sq + 1.0f);
				if(intensity > max_scent_intensity)
				{
					max_scent_intensity = intensity;
					strongest_scent_dir = offset;
				}
			}
			if(dist_sq < detection_radius_sq && dist_sq < min_feeder_prey_dist_sq && is_visible(offset, dist_sq))
			{
				min_feeder_prey_dist_sq = dist_sq;
				closest_feeder_prey_pos = other_pos;
				found_feeder_prey = true;
			}
		});
		world.m_predator_grid.visit(agent_pos, detection_radius, [&](const std::size_t i) {
			const Agent& other = world.m_agents[i];
			if(&other == this) return;
			const glm::vec3 other_pos = world.sensed_agent_transform(i)[3];
			const glm::vec3 offset = other_pos - agent_pos;
			const float dist_sq = glm::length2(offset);
			if(dist_sq < DENSITY_RADIUS_SQ) ++hostile_agent_count;
			if(dist_sq < detection_radius_sq && dist_sq < min_predator_prey_dist_sq && is_visible(offset, dist_sq))
			{
				min_predator_prey_dist_sq = dist_sq;
				closest_predator_prey_pos = other_pos;
				found_predator_prey = true;
			}
		});

		if(found_feeder_prey)
		{
			closest_threat_pos = closest_feeder_prey_pos;
			min_threat_dist_sq = min_feeder_prey_dist_sq;
			found_threat = true;
		}
		else if(found_predator_prey)
		{
			closest_threat_pos = closest_predator_prey_pos;
			min_threat_dist_sq = min_predator_prey_dist_sq;
			found_threat = true;
		}
	}

	if(found_food) m_food_pos = closest_food_pos;
	if(found_follow) m_follow_agent_pos = closest_follow_pos;
	if(found_threat) m_threat_pos = closest_threat_pos;

	if(found_food)
	{
		glm::vec3 direction = closest_food_pos - agent_pos;
		if(glm::length2(direction) > 1e-6f) direction = glm::normalize(direction);
		else direction = glm::vec3(0.0f);
		values[perception::FOOD_DIRECTION] = direction.x;
		values[perception::FOOD_DIRECTION + 1] = direction.y;
		values[perception::FOOD_DIRECTION + 2] = direction.z;
		values[perception::FOOD_DISTANCE] = std::sqrt(min_food_dist_sq);
	}
	if(found_threat)
	{
		glm::vec3 direction = closest_threat_pos - agent_pos;
		if(glm::length2(direction) > 1e-6f) direction = glm::normalize(direction);
		else direction = glm::vec3(0.0f);
		values[perception::THREAT_DIRECTION] = direction.x;
		values[perception::THREAT_DIRECTION + 1] = direction.y;
		values[perception::THREAT_DIRECTION + 2] = direction.z;
		values[perception::THREAT_DISTANCE] = std::sqrt(min_threat_dist_sq);
		values[perception::THREAT_PROXIMITY] = 1.0f - std::clamp(
			values[perception::THREAT_DISTANCE] / detection_radius, 0.0f, 1.0f);
	}
	if(found_follow)
	{
		glm::vec3 direction = closest_follow_pos - agent_pos;
		if(glm::length2(direction) > 1e-6f) direction = glm::normalize(direction);
		else direction = glm::vec3(0.0f);
		values[perception::FRIEND_DIRECTION] = direction.x;
		values[perception::FRIEND_DIRECTION + 1] = direction.y;
		values[perception::FRIEND_DIRECTION + 2] = direction.z;
		values[perception::FRIEND_DISTANCE] = std::sqrt(min_follow_dist_sq);
	}
	if(max_scent_intensity > 0.0f)
	{
		if(glm::length2(strongest_scent_dir) > 1e-6f) strongest_scent_dir = glm::normalize(strongest_scent_dir);
		values[perception::SCENT_DIRECTION] = strongest_scent_dir.x;
		values[perception::SCENT_DIRECTION + 1] = strongest_scent_dir.y;
		values[perception::SCENT_DIRECTION + 2] = strongest_scent_dir.z;
		values[perception::SCENT_INTENSITY] = std::min(max_scent_intensity, 1.0f);
	}

	glm::vec3 loudest_sound_dir(0.0f);
	float max_sound_intensity = 0.0f;
	for(const SoundEvent& sound : world.m_audible_sound_events)
	{
		const glm::vec3 offset = sound.position - agent_pos;
		const float dist_sq = glm::length2(offset);
		if(dist_sq < MAX_SOUND_RADIUS_SQ)
		{
			const float perceived_intensity = sound.intensity / (dist_sq + 1.0f);
			if(perceived_intensity > max_sound_intensity)
			{
				max_sound_intensity = perceived_intensity;
				loudest_sound_dir = offset;
			}
		}
	}
	if(max_sound_intensity > 0.0f)
	{
		if(glm::length2(loudest_sound_dir) > 1e-6f) loudest_sound_dir = glm::normalize(loudest_sound_dir);
		values[perception::SOUND_DIRECTION] = loudest_sound_dir.x;
		values[perception::SOUND_DIRECTION + 1] = loudest_sound_dir.y;
		values[perception::SOUND_DIRECTION + 2] = loudest_sound_dir.z;
		values[perception::SOUND_INTENSITY] = std::min(max_sound_intensity, 1.0f);
	}

	glm::vec3 boundary_avoidance(0.0f);
	const float half_world_size = world.m_world_size / 2.0f;
	if(agent_pos.x > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance.x -= 1.0f;
	if(agent_pos.x < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance.x += 1.0f;
	if(agent_pos.y > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance.y -= 1.0f;
	if(agent_pos.y < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance.y += 1.0f;
	if(agent_pos.z > half_world_size - BOUNDARY_THRESHOLD) boundary_avoidance.z -= 1.0f;
	if(agent_pos.z < -half_world_size + BOUNDARY_THRESHOLD) boundary_avoidance.z += 1.0f;
	if(glm::length2(boundary_avoidance) > 1e-6f) boundary_avoidance = glm::normalize(boundary_avoidance);
	values[perception::BOUNDARY_DIRECTION] = boundary_avoidance.x;
	values[perception::BOUNDARY_DIRECTION + 1] = boundary_avoidance.y;
	values[perception::BOUNDARY_DIRECTION + 2] = boundary_avoidance.z;

	const float nearest_boundary = std::min({
		half_world_size - agent_pos.x, half_world_size + agent_pos.x,
		half_world_size - agent_pos.y, half_world_size + agent_pos.y,
		half_world_size - agent_pos.z, half_world_size + agent_pos.z});
	values[perception::BOUNDARY_PROXIMITY] = 1.0f - std::clamp(
		nearest_boundary / BOUNDARY_AWARENESS_DISTANCE, 0.0f, 1.0f);
	values[perception::FRIENDLY_DENSITY] = std::min(friendly_agent_count / 10.0f, 1.0f);
	values[perception::HOSTILE_DENSITY] = std::min(hostile_agent_count / 10.0f, 1.0f);
	values[perception::FOOD_DENSITY] = std::min(food_density_count / 10.0f, 1.0f);

	const float max_energy = (m_type == AgentType::PREDATOR) ? PREDATOR_ENERGY_TO_REPRODUCE : FEEDER_ENERGY_TO_REPRODUCE;
	values[perception::ENERGY] = std::clamp(m_energy / max_energy, 0.0f, 1.0f);
	values[perception::AGE] = static_cast<float>(m_age) / static_cast<float>(MAX_AGE);
	values[perception::VELOCITY] = m_velocity.x;
	values[perception::VELOCITY + 1] = m_velocity.y;
	values[perception::VELOCITY + 2] = m_velocity.z;
	values[perception::HEADING] = agent_fwd.x;
	values[perception::HEADING + 1] = agent_fwd.y;
	values[perception::HEADING + 2] = agent_fwd.z;
	values[perception::ENERGY_TREND] = std::clamp(m_energy_trend, -1.0f, 1.0f);
	return values;
}

// Rule-based teacher for the high-level state. The state brain learns from
// these transitions; richer continuous inputs are appended in train_state_brain().
void Agent::update_state_transitions(const Simulation& /* world */)
{
	State next_state;
	if(m_type == AgentType::FEEDER)
	{
		if(m_threat_pos != glm::vec3(0.0f)) next_state = State::FLEE;
		else if(m_energy < FEEDER_ENERGY_TO_REPRODUCE * 0.9f && m_food_pos != glm::vec3(0.0f))
			next_state = State::HUNT;
		else next_state = State::ROAM;
	}
	else
	{
		if(m_threat_pos != glm::vec3(0.0f)) next_state = State::HUNT;
		else if(m_food_pos != glm::vec3(0.0f) && m_energy < PREDATOR_ENERGY_TO_REPRODUCE)
			next_state = State::HUNT;
		else next_state = State::ROAM;
	}
	m_state = next_state;
}

// Teacher-forcing for the movement brain using the new directional sensor approach.
void Agent::prepare_brain_target(std::vector<float>& perception_vector)
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
	glm::vec3 boundary_avoidance_vec(
		perception_vector[perception::BOUNDARY_DIRECTION],
		perception_vector[perception::BOUNDARY_DIRECTION + 1],
		perception_vector[perception::BOUNDARY_DIRECTION + 2]);
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

	// The goal channels describe the exact teacher goal, including boundary
	// avoidance, rather than a potentially conflicting independent target.
	const glm::vec3 goal_direction = glm::length2(desired_direction) > 1e-6f
		? glm::normalize(desired_direction) : agent_fwd;
	perception_vector[perception::GOAL_DIRECTION] = goal_direction.x;
	perception_vector[perception::GOAL_DIRECTION + 1] = goal_direction.y;
	perception_vector[perception::GOAL_DIRECTION + 2] = goal_direction.z;
	perception_vector[perception::GOAL_ALIGNMENT] = std::clamp(
		glm::dot(agent_fwd, goal_direction), -1.0f, 1.0f);

	// --- Create Target Parameter Vector for the RNN ---
	std::vector<float> target_params(7);
	target_params[0] = desired_translation.x;
	target_params[1] = desired_translation.y;
	target_params[2] = desired_translation.z;
	target_params[3] = desired_rotation.x;
	target_params[4] = desired_rotation.y;
	target_params[5] = desired_rotation.z;
	target_params[6] = desired_rotation.w;

	m_movement_target = std::move(target_params);
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

	// Compact continuous context added in perception layout v2.
	input[perception::STATE_ENERGY_TREND] = m_perception[perception::ENERGY_TREND];
	input[perception::STATE_FOOD_DENSITY] = m_perception[perception::FOOD_DENSITY];
	input[perception::STATE_BOUNDARY_PROXIMITY] = m_perception[perception::BOUNDARY_PROXIMITY];
	input[perception::STATE_GOAL_ALIGNMENT] = m_perception[perception::GOAL_ALIGNMENT];
	input[perception::STATE_SPEED] = std::clamp(glm::length(m_velocity), 0.0f, 1.0f);

	// 2. One-hot target vector for the current "correct" state (teacher).
	const size_t target_state_id = static_cast<size_t>(m_state);
	const auto& target_state_vec = state_tokenizer().find_vector(target_state_id);

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
	child_brain.mutate(MUTATION_RATE, MUTATION_AMOUNT, m_rng);

	// Child spawns near parent's location with a small random offset to prevent clumping
	glm::mat4 child_transform = m_transform;
    std::uniform_real_distribution<float> offset_dist(-1.0f, 1.0f);
    glm::vec3 offset(offset_dist(m_rng), offset_dist(m_rng), offset_dist(m_rng));
    child_transform = glm::translate(child_transform, offset * 2.0f); // Offset by up to 2 units

	// Create the new agent. Learning rate is randomized by passing -1.
	// The child's state brain is also randomized.
	Agent child(std::move(child_brain), child_transform, m_type, -1.0f, child_energy,
		state_hidden_dim(m_brain_tier), m_rng(), m_brain_tier);
	
	// Mutate the state brain as well
	// NOTE: lsRNN does not have a mutate function, so the child gets a new random one.
	// This is a valid evolutionary strategy.
	
	return child;
}

state_checksum::type Agent::checksum() const
{
	auto hash = state_checksum::initial;
	state_checksum::add(hash, m_brain.checksum());
	state_checksum::add(hash, m_state_brain.checksum());
	for(const glm::mat4* transform : {&m_transform, &m_prev_transform})
	{
		for(int column = 0; column < 4; ++column)
			for(int row = 0; row < 4; ++row)
				state_checksum::add(hash, (*transform)[column][row]);
	}
	for(const float value : {m_energy, m_energy_decay_rate, m_learning_rate,
		m_energy_trend, m_previous_tick_energy, m_velocity.x, m_velocity.y, m_velocity.z})
		state_checksum::add(hash, value);
	state_checksum::add(hash, static_cast<std::uint64_t>(m_age));
	state_checksum::add(hash, static_cast<std::uint64_t>(m_state));
	state_checksum::add(hash, static_cast<std::uint64_t>(m_type));
	state_checksum::add(hash, static_cast<std::uint64_t>(m_brain_tier));
	state_checksum::add(hash, m_state_brain_hidden_dim);
	state_checksum::add(hash, m_gradient_accumulation_step);
	return hash;
}
