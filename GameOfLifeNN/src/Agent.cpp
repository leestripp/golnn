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
static std::uniform_real_distribution<float> lr_dist(0.0005f, 0.01f);
static std::uniform_real_distribution<float> mutation_chance_dist(0.0f, 1.0f);

// Static tokenizer for agent states.
static lsVecTokenizer state_tokenizer(Agent::NUM_STATES);
static bool tokenizer_initialized = false;

static void initialize_tokenizer()
{
    if(!tokenizer_initialized)
	{
        state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::IDLE));
        state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::ROAM));
        state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::HUNT));
        state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::FLEE));
        tokenizer_initialized = true;
    }
}


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
    m_energy_decay_rate(FEEDER_ENERGY_DECAY),
    m_state_brain(STATE_BRAIN_INPUT_DIM, STATE_BRAIN_HIDDEN_DIM, NUM_STATES, Task::CLASSIFICATION),
    m_last_state(State::IDLE)
{
    initialize_tokenizer(); // Ensure tokenizer is set up

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
    if (m_energy < 0.0f) {
        if (m_type == AgentType::PREDATOR) {
            m_energy = INITIAL_ENERGY * 1.5f;
        } else {
            m_energy = INITIAL_ENERGY;
        }
    }
    m_last_energy = m_energy; // Initialize energy tracker

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

// Main update loop for the agent
void Agent::update(const Simulation& world) {
    // 1. Basic housekeeping: age and decay energy
    m_age++;
    m_energy -= m_energy_decay_rate;
    if (isDead()) return;

    // --- Reward Logic for successful eating ---
    // If the agent was hunting last tick and its energy is higher now than at the start
    // of the last tick, it means it ate. We reward this by boosting the learning rate.
    float effective_learning_rate = m_learning_rate;
    if (m_last_state == State::HUNT && m_energy > m_last_energy) {
        effective_learning_rate *= 5.0f; // Reward: learn 5x faster
    }
    m_last_energy = m_energy; // Update energy tracker for next tick's comparison

    m_prev_transform = m_transform;

    // 2. Perceive the world, build the movement brain input vector, and update internal world model
    std::vector<float> perception_vector = perceive(world);

    // 3. Determine the "correct" state based on rules (Teacher)
    update_state_transitions(world);
    m_last_state = m_state; // Update state tracker for next tick's reward check

    // 4. Train the state-selection brain to predict the correct state
    train_state_brain(world);

    // 5. Train the movement brain based on the correct state
    train_brain(world, perception_vector, effective_learning_rate);

    // 6. Get the next action from the movement brain and update transform
    m_transform = m_brain.process_step(perception_vector);

    // 7. Enforce world boundaries
    glm::vec3 current_pos = m_transform[3];
    float half_world_size = world.m_world_size / 2.0f;
    current_pos.x = std::clamp(current_pos.x, -half_world_size, half_world_size);
    current_pos.y = std::clamp(current_pos.y, -half_world_size, half_world_size);
    current_pos.z = std::clamp(current_pos.z, -half_world_size, half_world_size);
    m_transform[3] = glm::vec4(current_pos, 1.0f);
    m_brain.set_state(m_transform); // Keep brain state in sync after clamping

    // 8. Update color based on energy
    float max_energy = (m_type == AgentType::PREDATOR) ? PREDATOR_ENERGY_TO_REPRODUCE : FEEDER_ENERGY_TO_REPRODUCE;
    float energy_factor = std::clamp(m_energy / max_energy, 0.1f, 1.0f); // Clamp to avoid being invisible
    m_color = m_base_color * energy_factor;
}

// Gathers sensory information, prepares the input vector for the movement RNN,
// and updates the agent's internal world model (m_food_pos, m_threat_pos, etc.).
std::vector<float> Agent::perceive(const Simulation& world) {
    // Reset internal world model
    m_food_pos = glm::vec3(0.0f);
    m_follow_agent_pos = glm::vec3(0.0f);
    m_threat_pos = glm::vec3(0.0f);

    std::vector<float> perception(13, 0.0f); // 4 for food, 4 for follow, 4 for threat, 1 for energy

    glm::vec3 agent_pos = m_transform[3];
    glm::mat3 agent_rot_inv = glm::inverse(glm::mat3(m_transform));

    // --- Find Targets ---
    glm::vec3 closest_food_pos(0.0f);
    glm::vec3 closest_follow_pos(0.0f);
    glm::vec3 closest_threat_pos(0.0f);
    float min_food_dist_sq = std::numeric_limits<float>::max();
    float min_follow_dist_sq = ROAM_FOLLOW_RADIUS_SQ;
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

    // Find closest food (for feeders)
    if (m_type == AgentType::FEEDER) {
        for (const auto& food : world.m_food) {
            float dist_sq = glm::distance2(agent_pos, food.position);
            if (dist_sq < FEEDER_FOOD_DETECTION_RADIUS_SQ && dist_sq < min_food_dist_sq) {
                min_food_dist_sq = dist_sq;
                closest_food_pos = food.position;
                found_food = true;
            }
        }
    }

    // Find closest agents (threat, follow, prey)
    for (const auto& other : world.m_agents) {
        if (&other == this) continue;
        glm::vec3 other_pos = other.m_transform[3];
        float dist_sq = glm::distance2(agent_pos, other_pos);

        if (m_type == AgentType::FEEDER) {
            // Feeder looks for predators (threat) and other feeders (follow)
            if (other.m_type == AgentType::PREDATOR && dist_sq < PREDATOR_DETECTION_RADIUS_SQ) {
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
            if (dist_sq < PREDATOR_HUNT_RADIUS_SQ) {
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
            found_threat = true;
        } else if (found_predator_prey) {
            closest_threat_pos = closest_predator_prey_pos;
            found_threat = true;
        }
    }

    // --- Update Internal World Model ---
    if (found_food) m_food_pos = closest_food_pos;
    if (found_follow) m_follow_agent_pos = closest_follow_pos;
    if (found_threat) m_threat_pos = closest_threat_pos;

    // --- Fill Perception Vector for Movement Brain (Direction & Distance) ---
    // Food
    if (found_food) {
        glm::vec3 dir_to_food = closest_food_pos - agent_pos;
        float dist = glm::length(dir_to_food);
        if (dist > 1e-6f) {
            glm::vec3 local_dir = agent_rot_inv * glm::normalize(dir_to_food);
            perception[0] = local_dir.x;
            perception[1] = local_dir.y;
            perception[2] = local_dir.z;
        }
        perception[3] = dist;
    }

    // Follow Agent
    if (found_follow) {
        glm::vec3 dir_to_follow = closest_follow_pos - agent_pos;
        float dist = glm::length(dir_to_follow);
        if (dist > 1e-6f) {
            glm::vec3 local_dir = agent_rot_inv * glm::normalize(dir_to_follow);
            perception[4] = local_dir.x;
            perception[5] = local_dir.y;
            perception[6] = local_dir.z;
        }
        perception[7] = dist;
    }

    // Threat/Prey Agent
    if (found_threat) {
        glm::vec3 dir_to_threat = closest_threat_pos - agent_pos;
        float dist = glm::length(dir_to_threat);
        if (dist > 1e-6f) {
            glm::vec3 local_dir = agent_rot_inv * glm::normalize(dir_to_threat);
            perception[8] = local_dir.x;
            perception[9] = local_dir.y;
            perception[10] = local_dir.z;
        }
        perception[11] = dist;
    }

    // Own Energy
    perception[12] = m_energy / 100.0f; // Normalize energy

    return perception;
}

// The "Teacher" - sets the correct state based on the agent's internal world model.
void Agent::update_state_transitions(const Simulation& /* world */)
{
    // Start in IDLE mode
    State next_state = State::IDLE;
	
    if(m_type == AgentType::FEEDER)
	{
        if(m_energy < INITIAL_ENERGY)
		{
			// Hungry
			if(m_threat_pos == glm::vec3(0.0f))
			{
				// HUNT
				next_state = State::HUNT;
			} else
			{
				// ROAM
				next_state = State::ROAM;
			}
		} else
		{
			// Not hungry
			if(m_threat_pos == glm::vec3(0.0f))
			{
				// IDLE
				next_state = State::IDLE;
			} else
			{
				// FLEE
				next_state = State::FLEE;
			}
		}
		// Prioitise HUNT
        if(m_energy < INITIAL_ENERGY * 0.75f)
		{
			// Fallback to HUNT
			next_state = State::HUNT;
		}
    } else
	{
        // Predator Logic
        // Predators are always hunting if they see prey.
        if(m_threat_pos != glm::vec3(0.0f))
		{
            next_state = State::HUNT;
        } else {
            next_state = State::ROAM; // Otherwise, roam to find prey.
        }
    }
    m_state = next_state;
}


// Teacher-forcing for the movement brain
void Agent::train_brain(const Simulation& /* world */, const std::vector<float>& perception_vector, float learning_rate) {
    glm::vec3 target_pos(0.0f);
    bool has_target = false;

    // Determine target based on state
    switch (m_state) {
        case State::HUNT:
            target_pos = (m_type == AgentType::FEEDER) ? m_food_pos : m_threat_pos;
            has_target = true;
            break;
        case State::FLEE:
            target_pos = m_threat_pos;
            has_target = true;
            break;
        case State::ROAM:
            target_pos = m_follow_agent_pos;
            has_target = (target_pos != glm::vec3(0.0f));
            break;
        case State::IDLE:
        default:
            break; // No specific target
    }

    glm::vec3 agent_pos = m_transform[3];
    glm::vec3 agent_fwd = glm::normalize(glm::vec3(m_transform[2]));

    // --- Calculate Ideal Delta Transform ---
    glm::vec3 desired_translation(0.0f);
    glm::quat desired_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity

    if (has_target && target_pos != glm::vec3(0.0f)) {
        glm::vec3 dir_to_target = target_pos - agent_pos;
        if (glm::length2(dir_to_target) > 1e-6f) {
            dir_to_target = glm::normalize(dir_to_target);
            if (m_state == State::FLEE) {
                dir_to_target = -dir_to_target; // Flee in the opposite direction
            }
            desired_translation = dir_to_target * MAX_SPEED;
            
            // Smoothly rotate towards the target direction
            glm::quat target_rot = glm::rotation(agent_fwd, dir_to_target);
            desired_rotation = glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), target_rot, ROTATION_LERP_FACTOR);
        }

    } else { // IDLE or ROAM without a follow target
        // Just move forward slowly
        desired_translation = agent_fwd * (MAX_SPEED * 0.2f);
    }

    // --- Create Target Parameter Vector ---
    std::vector<float> target_params(7);
    target_params[0] = desired_translation.x;
    target_params[1] = desired_translation.y;
    target_params[2] = desired_translation.z;
    target_params[3] = desired_rotation.x;
    target_params[4] = desired_rotation.y;
    target_params[5] = desired_rotation.z;
    target_params[6] = desired_rotation.w;

    // --- Train the Brain ---
    m_brain.train_step(perception_vector, target_params, learning_rate, ROTATION_WEIGHT);
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
    m_state_brain.backpropagate(inputs_sequence, outputs_sequence, expected_sequence, m_learning_rate);
}

bool Agent::isDead() const {
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
