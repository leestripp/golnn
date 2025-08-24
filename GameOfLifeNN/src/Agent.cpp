#include "Agent.h"
#include "Simulation.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include <iostream>
#include <algorithm>
#include <limits>

// Agent constructor
Agent::Agent(ls3DRNN brain, const glm::mat4& initial_transform)
    : m_brain(std::move(brain)),
      m_transform(initial_transform),
      m_prev_transform(initial_transform),
      m_energy(INITIAL_ENERGY),
      m_age(0),
      m_state(State::IDLE),
      m_color({0.7f, 0.7f, 0.7f}), // Grey for IDLE
      m_food_pos(0.0f),
      m_follow_agent_pos(0.0f)
{
    // Synchronize the brain's internal state with the agent's initial transform.
    // This is crucial for the RNN's stateful processing to be correct.
    m_brain.set_state(m_transform);
}

// This new function handles the transition between states.
// It also determines the targets for the current state (food_pos, follow_agent_pos)
void Agent::update_state_transitions(const Simulation& world)
{
    glm::vec3 agent_pos = m_transform[3];
    
    // 1. Reset target positions. They will be set below if a target is found.
    m_food_pos = glm::vec3(0.0f);
    m_follow_agent_pos = glm::vec3(0.0f);
	
    // 2. Highest priority: HUNT if energy is low.
    if (m_energy < HUNTING_ENERGY_THRESHOLD)
    {
        // Try to find food
        float min_dist_sq = std::numeric_limits<float>::max();
        bool food_found = false;
        for(const auto& food : world.m_food)
		{
            float dist_sq = glm::distance2(agent_pos, food.position);
            if(dist_sq < min_dist_sq)
			{
                min_dist_sq = dist_sq;
                m_food_pos = food.position;
                food_found = true;
            }
        }
		
        if(food_found)
		{
            m_state = State::HUNT;
            m_color = {1.0f, 0.8f, 0.2f}; // Yellow for HUNT
            return; // State decided
        }
        // If no food is found while hungry, fall through to ROAM to find other agents who might be near food.
    }
	
    // 3. If not hunting, try to ROAM (follow another agent)
    float min_agent_dist_sq = ROAM_FOLLOW_RADIUS_SQ;
    bool agent_found = false;
    for(const auto& other_agent : world.m_agents)
	{
        if (&other_agent == this) continue; // Don't follow self
        
        float dist_sq = glm::distance2(agent_pos, glm::vec3(other_agent.m_transform[3]));
        if(dist_sq < min_agent_dist_sq)
		{
            min_agent_dist_sq = dist_sq;
            m_follow_agent_pos = other_agent.m_transform[3];
            agent_found = true;
        }
    }
	
    if(agent_found)
	{
        m_state = State::ROAM;
        m_color = {0.2f, 0.5f, 1.0f}; // Blue for ROAM
    } else
	{
        // 4. If no agents nearby, become IDLE
		m_state = State::IDLE;
        m_color = {0.7f, 0.7f, 0.7f}; // Grey for IDLE
    }
}


// This function calculates the "correct" action based on the current state and trains the neural network.
void Agent::train_brain(const Simulation& /* world */, const std::vector<float>& perception_vector)
{
    std::vector<float> target_params(7, 0.0f); // Default target is to do nothing
    glm::vec3 agent_pos = m_transform[3];
	
    // Helper lambda to generate target parameters for moving towards a point
    auto calculate_target_params = [&](const glm::vec3& target_pos) {
        // 1. Calculate Target Translation Delta: Move towards the target
        glm::vec3 target_translation_delta = glm::normalize(target_pos - agent_pos) * MAX_SPEED;
        
        // 2. Calculate Target Rotation Delta: Turn to face the target
        glm::vec3 current_forward = glm::normalize(glm::vec3(m_transform[2])); // Agent's local Z-axis
        glm::vec3 direction_to_target = glm::normalize(target_pos - agent_pos);
        
        // Avoid undefined rotation if already at the target
        if (glm::length2(direction_to_target) < 1e-6) {
            return std::vector<float>(7, 0.0f);
        }
        
        glm::quat target_rotation_delta = glm::rotation(current_forward, direction_to_target);
        
        // 3. Populate the target parameters vector
        std::vector<float> params(7);
        params[0] = target_translation_delta.x;
        params[1] = target_translation_delta.y;
        params[2] = target_translation_delta.z;
        params[3] = target_rotation_delta.x;
        params[4] = target_rotation_delta.y;
        params[5] = target_rotation_delta.z;
        params[6] = target_rotation_delta.w;
        return params;
    };
	
    switch(m_state)
	{
        case State::HUNT:
            if (m_food_pos != glm::vec3(0.0f)) target_params = calculate_target_params(m_food_pos);
            break;
        case State::ROAM:
            if (m_follow_agent_pos != glm::vec3(0.0f)) target_params = calculate_target_params(m_follow_agent_pos);
            break;
        case State::IDLE:
            // Default target is to stop, which is what we want.
            break;
    }
	
    // Train the Brain
    // Teach the brain to output our desired `target_params` when it receives `perception_vector`.
    m_brain.train_step(perception_vector, target_params, LEARNING_RATE, ROTATION_WEIGHT);
}

// The main update function for an agent, called each simulation tick
void Agent::update(const Simulation& world)
{
	m_age++;
	m_energy -= ENERGY_DECAY_PER_TICK;
	
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
}


// Check if the agent should be removed
bool Agent::isDead() const {
    return m_energy <= 0.0f || m_age > MAX_AGE;
}

// Check if the agent can reproduce
bool Agent::canReproduce() const {
    return m_energy >= ENERGY_TO_REPRODUCE;
}

// Create a new agent (offspring)
Agent Agent::reproduce()
{
    m_energy /= 2.0f; // Parent gives half its energy to the child
	
    ls3DRNN child_brain = m_brain;
    child_brain.mutate(0.01f, 0.0001f);
	
    return Agent(std::move(child_brain), m_transform);
}


// Collects sensory information from the world to feed into the RNN
std::vector<float> Agent::perceive(const Simulation& /* world */) const {
    // These inputs must match the size defined in Simulation.cpp (perception_vector_size = 7)
    std::vector<float> perception_vector;
    perception_vector.reserve(7);

    // Input 1: Food Position (world space). Zero if not in HUNT state.
    perception_vector.push_back(m_food_pos.x);
    perception_vector.push_back(m_food_pos.y);
    perception_vector.push_back(m_food_pos.z);

    // Input 2: Follow Agent Position (world space). Zero if not in ROAM state.
    perception_vector.push_back(m_follow_agent_pos.x);
    perception_vector.push_back(m_follow_agent_pos.y);
    perception_vector.push_back(m_follow_agent_pos.z);

    // Input 3: Own Energy
    perception_vector.push_back(m_energy);

    return perception_vector;
}
