#include "Agent.h"
#include "Simulation.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <iostream>
#include <algorithm>
#include <limits>
#include <random>

// Random number generator for agent-specific properties like learning rate
static std::random_device rd;
static std::mt19937 agent_gen(rd());
static std::uniform_real_distribution<float> lr_dist(0.0001f, 0.01f);
static std::uniform_real_distribution<float> mutation_chance_dist(0.0f, 1.0f);


// Agent constructor
Agent::Agent(ls3DRNN brain, const glm::mat4& initial_transform, AgentType type, float learning_rate, float initial_energy) :
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
    m_energy_decay_rate(FEEDER_ENERGY_DECAY)
{
    // Assign learning rate: random if not provided, otherwise use the given value.
    if (learning_rate < 0.0f) {
        m_learning_rate = lr_dist(agent_gen);
    } else {
        m_learning_rate = learning_rate;
    }

    // If initial_energy was not provided (i.e., it's a newly spawned agent, not an offspring),
    // set the energy based on the agent's type.
    if (m_energy < 0.0f) {
        if (m_type == AgentType::PREDATOR) {
            m_energy = INITIAL_ENERGY * 1.5f;
        } else {
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
	}
    else // FEEDER
    {
        m_base_color = {0.0f, 0.0f, 1.0f}; // Deep Blue
    }
}

// This new function handles the transition between states.
// It also determines the targets for the current state (food_pos, follow_agent_pos)
void Agent::update_state_transitions(const Simulation& world)
{
	glm::vec3 agent_pos = m_transform[3];
	
	// 1. Reset target positions.
	m_food_pos = glm::vec3(0.0f);
	m_follow_agent_pos = glm::vec3(0.0f);
	m_threat_pos = glm::vec3(0.0f);

	if (m_type == AgentType::FEEDER)
	{
		// --- FEEDER LOGIC ---
		// 1. Highest priority: FLEE from predators
		float min_predator_dist_sq = PREDATOR_DETECTION_RADIUS_SQ;
		bool predator_found = false;
		for (const auto& other_agent : world.m_agents) {
			if (other_agent.m_type == AgentType::PREDATOR) {
				float dist_sq = glm::distance2(agent_pos, glm::vec3(other_agent.m_transform[3]));
				if (dist_sq < min_predator_dist_sq) {
					min_predator_dist_sq = dist_sq;
					m_threat_pos = other_agent.m_transform[3];
					predator_found = true;
				}
			}
		}

		if (predator_found) {
			m_state = State::FLEE;
			return; // State decided
		}

		// 2. Next priority: HUNT for food if energy is not sufficient for reproduction.
		// This is the key change to prevent population explosion. Agents stop actively
		// seeking food once they have enough energy to reproduce.
		if (m_energy < FEEDER_ENERGY_TO_REPRODUCE) {
			float min_dist_sq = std::numeric_limits<float>::max();
			bool food_found = false;
			for (const auto& food : world.m_food) {
				float dist_sq = glm::distance2(agent_pos, food.position);
				if (dist_sq < min_dist_sq) {
					min_dist_sq = dist_sq;
					m_food_pos = food.position;
					food_found = true;
				}
			}
			if (food_found) {
				m_state = State::HUNT;
				return; // State decided
			}
		}

		// 3. If not fleeing or hunting, try to ROAM (follow another feeder)
		float min_agent_dist_sq = ROAM_FOLLOW_RADIUS_SQ;
		bool agent_found = false;
		for (const auto& other_agent : world.m_agents) {
			if (&other_agent == this || other_agent.m_type == AgentType::PREDATOR) continue;
			
			float dist_sq = glm::distance2(agent_pos, glm::vec3(other_agent.m_transform[3]));
			if (dist_sq < min_agent_dist_sq) {
				min_agent_dist_sq = dist_sq;
				m_follow_agent_pos = other_agent.m_transform[3];
				agent_found = true;
			}
		}
		
		if (agent_found) {
			m_state = State::ROAM;
		} else {
			// 4. If no agents nearby, become IDLE
			m_state = State::IDLE;
		}
	}
	else // m_type == AgentType::PREDATOR
	{
		// --- PREDATOR LOGIC ---
		// 1. Highest priority: HUNT for feeders within perception range
		float min_prey_dist_sq = PREDATOR_HUNT_RADIUS_SQ; // FIX: Use finite hunt radius
		bool feeder_prey_found = false;
		glm::vec3 target_pos(0.0f);

		for (const auto& other_agent : world.m_agents) {
			if (other_agent.m_type == AgentType::FEEDER) {
				float dist_sq = glm::distance2(agent_pos, glm::vec3(other_agent.m_transform[3]));
				if (dist_sq < min_prey_dist_sq) {
					min_prey_dist_sq = dist_sq;
					target_pos = other_agent.m_transform[3];
					feeder_prey_found = true;
				}
			}
		}

		if (feeder_prey_found) {
			m_threat_pos = target_pos;
			m_state = State::HUNT;
			return; // State decided
		}

		// 2. If no feeders, hunt other predators within perception range
		min_prey_dist_sq = PREDATOR_HUNT_RADIUS_SQ; // FIX: Use finite hunt radius
		bool predator_prey_found = false;
		for (const auto& other_agent : world.m_agents) {
			// Can't hunt itself
			if (&other_agent == this) continue;

			if (other_agent.m_type == AgentType::PREDATOR) {
				float dist_sq = glm::distance2(agent_pos, glm::vec3(other_agent.m_transform[3]));
				if (dist_sq < min_prey_dist_sq) {
					min_prey_dist_sq = dist_sq;
					target_pos = other_agent.m_transform[3];
					predator_prey_found = true;
				}
			}
		}

		if (predator_prey_found) {
			m_threat_pos = target_pos;
			m_state = State::HUNT;
			return; // State decided
		}
		
		// 3. If no prey at all in range, become IDLE
		m_state = State::IDLE;
	}
}


// This function calculates the "correct" action based on the current state and trains the neural network.
void Agent::train_brain(const Simulation& /* world */, const std::vector<float>& perception_vector)
{
	std::vector<float> target_params(7, 0.0f); // Default target is to do nothing
	glm::vec3 agent_pos = m_transform[3];
 	
    // Helper lambda to generate target parameters for moving towards or away from a point
    auto calculate_target_params = [&](const glm::vec3& target_pos, bool flee = false){
        glm::vec3 direction_vec = target_pos - agent_pos;
        if (flee) {
            direction_vec = -direction_vec; // Invert direction to flee
        }

        // Avoid undefined rotation if already at the target or if direction is zero
        if (glm::length2(direction_vec) < 1e-6) {
            return std::vector<float>(7, 0.0f);
        }

        // 1. Calculate Target Translation Delta
        glm::vec3 target_translation_delta = glm::normalize(direction_vec) * MAX_SPEED;

        // 2. Calculate Target Rotation Delta
        glm::vec3 current_forward = glm::normalize(glm::vec3(m_transform[2]));
        glm::vec3 direction_to_target = glm::normalize(direction_vec);
		glm::quat target_rotation_delta = glm::rotation(current_forward, direction_to_target);
         
        // 3. Populate the target parameters vector
        // Explicitly construct the std::vector<float> to resolve the type deduction ambiguity.
        return std::vector<float>{
            target_translation_delta.x, target_translation_delta.y, target_translation_delta.z,
            target_rotation_delta.x, target_rotation_delta.y, target_rotation_delta.z, target_rotation_delta.w
        };
     };
 	
	switch(m_state)
 	{
         case State::HUNT:
            if (m_type == AgentType::FEEDER) {
                if (m_food_pos != glm::vec3(0.0f)) target_params = calculate_target_params(m_food_pos);
            } else { // PREDATOR
                if (m_threat_pos != glm::vec3(0.0f)) target_params = calculate_target_params(m_threat_pos);
            }
             break;
         case State::ROAM:
             if (m_follow_agent_pos != glm::vec3(0.0f)) target_params = calculate_target_params(m_follow_agent_pos);
            break;
        case State::FLEE:
            if (m_threat_pos != glm::vec3(0.0f)) target_params = calculate_target_params(m_threat_pos, true); // Flee!
             break;
         case State::IDLE:
             // Default target is to stop, which is what we want.
            break;
    }
	
    // Train the Brain
    // Teach the brain to output our desired `target_params` when it receives `perception_vector`.
    m_brain.train_step(perception_vector, target_params, m_learning_rate, ROTATION_WEIGHT);
}

// The main update function for an agent, called each simulation tick
void Agent::update(const Simulation& world)
{
	m_age++;
	m_energy -= m_energy_decay_rate; // Use type-specific energy decay
	
	m_prev_transform = m_transform;
	
	// 1. Update State: Decide current state and find targets.
    update_state_transitions(world);

	// 2. Perceive: Get inputs for the neural network based on the new state.
	std::vector<float> perception_vector = perceive(world);
	
    // 3. Train: Calculate the ideal action based on the state and train the brain.
    train_brain(world, perception_vector);

	// 4. Think: Process inputs through the NOW-TRAINED RNN.
	const glm::mat4& target_transform = m_brain.process_step(perception_vector);
	
	// 5. Act: Apply physical constraints to the brain's desired transform.
	glm::vec3 current_pos = m_transform[3];
	glm::vec3 target_pos = target_transform[3];

	glm::vec3 movement_vector = target_pos - current_pos;
	float movement_length = glm::length(movement_vector);
	
	if (movement_length > MAX_SPEED) {
		movement_vector = (movement_vector / movement_length) * MAX_SPEED; 
	}
	
	glm::vec3 new_pos = current_pos + movement_vector;
	float half_world_size = world.m_world_size / 2.0f;
	new_pos.x = std::clamp(new_pos.x, -half_world_size, half_world_size);
	new_pos.y = std::clamp(new_pos.y, -half_world_size, half_world_size);
	new_pos.z = std::clamp(new_pos.z, -half_world_size, half_world_size);
	
	glm::quat current_rot = glm::quat_cast(m_transform);
	glm::quat target_rot = glm::quat_cast(target_transform);
	glm::quat new_rot = glm::slerp(current_rot, target_rot, ROTATION_LERP_FACTOR);
	
	m_transform = glm::translate(glm::mat4(1.0f), new_pos) * glm::mat4_cast(new_rot);
	
	// CRITICAL: Synchronize the brain's internal state with the agent's actual state.
	m_brain.set_state(m_transform);

    // 6. Update Color based on energy
    // Bright = high energy, dark = low energy.
    // We use INITIAL_ENERGY as the baseline for "full brightness".
    float energy_factor = std::clamp(m_energy / INITIAL_ENERGY, 0.1f, 1.0f);
    m_color = m_base_color * energy_factor;
}


// Check if the agent should be removed
bool Agent::isDead() const {
    return m_energy <= 0.0f || m_age > MAX_AGE;
}

// Check if the agent can reproduce using type-specific energy requirements
bool Agent::canReproduce() const {
    if (m_type == AgentType::PREDATOR) {
        return m_energy >= PREDATOR_ENERGY_TO_REPRODUCE;
    }
    return m_energy >= FEEDER_ENERGY_TO_REPRODUCE;
}

// Create a new agent (offspring)
Agent Agent::reproduce()
{
    m_energy /= 2.0f; // Parent gives half its energy to the child
    float child_energy = m_energy; // Child gets the other half
	
    ls3DRNN child_brain = m_brain;
    child_brain.mutate(MUTATION_RATE, MUTATION_AMOUNT);
	
    // Mutate the learning rate as well
    if (mutation_chance_dist(agent_gen) < MUTATION_RATE) {
        // Mutate by up to 10% of the current value, either up or down
        std::uniform_real_distribution<float> lr_mutation_dist(-0.1f, 0.1f);
        float mutation = m_learning_rate * lr_mutation_dist(agent_gen);
        m_learning_rate += mutation;
    }
    // Clamp the learning rate to the valid range [0.0001, 0.01]
    float child_learning_rate = std::clamp(m_learning_rate, 0.0001f, 0.01f);
	
	// Child has the same type as the parent, the (potentially mutated) learning rate, and half the parent's energy
	return Agent(std::move(child_brain), m_transform, m_type, child_learning_rate, child_energy);
}


// Collects sensory information from the world to feed into the RNN
std::vector<float> Agent::perceive(const Simulation& /* world */) const {
    // New size: 4 (food) + 4 (follow) + 4 (threat) + 1 (energy) = 13
	std::vector<float> perception_vector;
	perception_vector.reserve(13);
	
     glm::vec3 agent_pos = m_transform[3];
     glm::mat3 world_to_local_rot = glm::inverse(glm::mat3(m_transform));
 
    // Helper lambda to add a perception vector (dir + dist)
    auto add_perception = [&](const glm::vec3& target_pos) {
        if (target_pos != glm::vec3(0.0f)) {
            glm::vec3 vec_to_target = target_pos - agent_pos;
            float dist = glm::length(vec_to_target);
            if (dist > 1e-6f) {
                glm::vec3 dir_world = vec_to_target / dist;
                glm::vec3 dir_local = world_to_local_rot * dir_world;
                perception_vector.push_back(dir_local.x);
                perception_vector.push_back(dir_local.y);
                perception_vector.push_back(dir_local.z);
                perception_vector.push_back(dist);
            } else {
                perception_vector.insert(perception_vector.end(), {0.0f, 0.0f, 0.0f, 0.0f});
            }
        } else {
            perception_vector.insert(perception_vector.end(), {0.0f, 0.0f, 0.0f, 0.0f});
        }
    };

     // --- Input 1: Food (Local Direction + Distance) ---
    // Only feeders perceive food
    if (m_type == AgentType::FEEDER) {
        add_perception(m_food_pos);
     } else
	 {
         perception_vector.insert(perception_vector.end(), {0.0f, 0.0f, 0.0f, 0.0f});
     }
 
     // --- Input 2: Follow Agent (Local Direction + Distance) ---
    // Only feeders follow other feeders
    if (m_type == AgentType::FEEDER) {
        add_perception(m_follow_agent_pos);
     } else
	 {
         perception_vector.insert(perception_vector.end(), {0.0f, 0.0f, 0.0f, 0.0f});
     }

    // --- Input 3: Threat/Prey (Local Direction + Distance) ---
    // Both types perceive this: feeders see predators, predators see feeders.
    add_perception(m_threat_pos);

    // --- Input 4: Own Energy ---
     perception_vector.push_back(m_energy);
 
	return perception_vector;
}
