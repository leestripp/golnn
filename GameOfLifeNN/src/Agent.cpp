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
static std::mt19937 agent_gen(rd()); // FIX: Corrected typo from mt19137
static std::uniform_real_distribution<float> lr_dist(0.0001f, 0.01f);
static std::uniform_real_distribution<float> mutation_chance_dist(0.0f, 1.0f);

// Static tokenizer for agent states.
static lsVecTokenizer state_tokenizer(Agent::NUM_STATES);
static bool tokenizer_initialized = false;

static void initialize_tokenizer() {
    if (!tokenizer_initialized) {
        state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::IDLE));
        state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::ROAM));
        state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::HUNT));
        state_tokenizer.add_token_onehot(static_cast<size_t>(Agent::State::FLEE));
        tokenizer_initialized = true;
    }
}


// Agent constructor
// FIX: Reordered initializer list to match Agent.h declaration order and remove warnings.
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

    // 2. Perceive the world and build the input vector for the brain
    std::vector<float> perception_vector = perceive(world);

    // 3. Determine the "correct" state based on rules (Teacher)
    update_state_transitions(world);
    m_last_state = m_state; // Update state tracker for next tick's reward check

    // 4. Train the state-selection brain to predict the correct state
    train_state_brain();

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

// Gathers sensory information and prepares the input vector for the RNNs.
std::vector<float> Agent::perceive(const Simulation& world) const {
    std::vector<float> perception(13, 0.0f); // 4 for food, 4 for follow, 4 for threat, 1 for energy

    glm::vec3 agent_pos = m_transform[3];
    glm::mat3 agent_rot_inv = glm::inverse(glm::mat3(m_transform));

    // --- Find Targets ---
    // These are world-space positions
    glm::vec3 closest_food_pos(0.0f);
    glm::vec3 closest_follow_pos(0.0f);
    glm::vec3 closest_threat_pos(0.0f);
    float min_food_dist_sq = std::numeric_limits<float>::max();
    float min_follow_dist_sq = ROAM_FOLLOW_RADIUS_SQ;
    float min_threat_dist_sq = std::numeric_limits<float>::max();

    // Find closest food
    for (const auto& food : world.m_food) {
        float dist_sq = glm::distance2(agent_pos, food.position);
        if (dist_sq < min_food_dist_sq) {
            min_food_dist_sq = dist_sq;
            closest_food_pos = food.position;
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
                }
            } else if (other.m_type == AgentType::FEEDER && dist_sq < min_follow_dist_sq) {
                min_follow_dist_sq = dist_sq;
                closest_follow_pos = other_pos;
            }
        } else { // PREDATOR
            // Predator looks for any other agent as prey (threat)
            if (dist_sq < PREDATOR_HUNT_RADIUS_SQ) {
                if (dist_sq < min_threat_dist_sq) {
                    min_threat_dist_sq = dist_sq;
                    closest_threat_pos = other_pos;
                }
            }
        }
    }

    // --- Fill Perception Vector (Direction & Distance) ---
    // Food
    if (min_food_dist_sq != std::numeric_limits<float>::max()) {
        glm::vec3 dir_to_food = closest_food_pos - agent_pos;
        float dist = glm::length(dir_to_food);
        if (dist > 1e-6f) { // Avoid normalization of zero vector
            glm::vec3 local_dir = agent_rot_inv * glm::normalize(dir_to_food);
            perception[0] = local_dir.x;
            perception[1] = local_dir.y;
            perception[2] = local_dir.z;
        }
        perception[3] = dist;
    }

    // Follow Agent
    if (min_follow_dist_sq != ROAM_FOLLOW_RADIUS_SQ) {
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
    if (min_threat_dist_sq != std::numeric_limits<float>::max()) {
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


// The "Teacher" - sets the correct state based on external conditions.
void Agent::update_state_transitions(const Simulation& world) {
    // Reset target positions
    m_food_pos = glm::vec3(0.0f);
    m_follow_agent_pos = glm::vec3(0.0f);
    m_threat_pos = glm::vec3(0.0f);

    glm::vec3 agent_pos = m_transform[3];
    State next_state = State::IDLE;

    if (m_type == AgentType::FEEDER) {
        // --- Feeder Logic ---
        // 1. Priority: Flee from predators
        Agent* closest_predator = nullptr;
        float min_pred_dist_sq = PREDATOR_DETECTION_RADIUS_SQ;
        for (const auto& other : world.m_agents) {
            if (other.m_type == AgentType::PREDATOR) {
                float dist_sq = glm::distance2(agent_pos, glm::vec3(other.m_transform[3]));
                if (dist_sq < min_pred_dist_sq) {
                    min_pred_dist_sq = dist_sq;
                    closest_predator = const_cast<Agent*>(&other);
                }
            }
        }
        if (closest_predator) {
            next_state = State::FLEE;
            m_threat_pos = closest_predator->m_transform[3];
        } else {
            // 2. If not fleeing, look for food if energy is low
            if (m_energy < FEEDER_ENERGY_TO_REPRODUCE * 0.8f) {
                const Food* closest_food = nullptr;
                float min_food_dist_sq = std::numeric_limits<float>::max();
                for (const auto& food : world.m_food) {
                    float dist_sq = glm::distance2(agent_pos, food.position);
                    if (dist_sq < min_food_dist_sq) {
                        min_food_dist_sq = dist_sq;
                        closest_food = &food;
                    }
                }
                if (closest_food) {
                    next_state = State::HUNT;
                    m_food_pos = closest_food->position;
                }
            }
            // 3. If no immediate threat or hunger, roam with friends
            if (next_state == State::IDLE) {
                Agent* closest_feeder = nullptr;
                float min_feeder_dist_sq = ROAM_FOLLOW_RADIUS_SQ;
                for (const auto& other : world.m_agents) {
                    if (&other != this && other.m_type == AgentType::FEEDER) {
                        float dist_sq = glm::distance2(agent_pos, glm::vec3(other.m_transform[3]));
                        if (dist_sq < min_feeder_dist_sq) {
                            min_feeder_dist_sq = dist_sq;
                            closest_feeder = const_cast<Agent*>(&other);
                        }
                    }
                }
                if (closest_feeder) {
                    next_state = State::ROAM;
                    m_follow_agent_pos = closest_feeder->m_transform[3];
                }
            }
        }
    } else { // PREDATOR
        // --- Predator Logic ---
        Agent* closest_prey = nullptr;
        float min_prey_dist_sq = PREDATOR_HUNT_RADIUS_SQ;
        // Prioritize feeders
        for (const auto& other : world.m_agents) {
            if (other.m_type == AgentType::FEEDER) {
                float dist_sq = glm::distance2(agent_pos, glm::vec3(other.m_transform[3]));
                if (dist_sq < min_prey_dist_sq) {
                    min_prey_dist_sq = dist_sq;
                    closest_prey = const_cast<Agent*>(&other);
                }
            }
        }
        // If no feeders, hunt other predators
        if (!closest_prey) {
            for (const auto& other : world.m_agents) {
                if (&other != this && other.m_type == AgentType::PREDATOR) {
                    float dist_sq = glm::distance2(agent_pos, glm::vec3(other.m_transform[3]));
                    if (dist_sq < min_prey_dist_sq) {
                        min_prey_dist_sq = dist_sq;
                        closest_prey = const_cast<Agent*>(&other);
                    }
                }
            }
        }

        if (closest_prey) {
            next_state = State::HUNT;
            m_threat_pos = closest_prey->m_transform[3];
        } else {
            next_state = State::ROAM; // Always be on the move
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
            target_pos = (m_food_pos != glm::vec3(0.0f)) ? m_food_pos : m_threat_pos;
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

    if (has_target) {
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
void Agent::train_state_brain()
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

    // Normalize distance based on perception radius. Anything beyond is considered "far".
    // For feeders, the detection radius is for predators. For predators, it's for their prey.
    float max_detection_radius = (m_type == AgentType::FEEDER) ? 
                                 sqrt(PREDATOR_DETECTION_RADIUS_SQ) : 
                                 sqrt(PREDATOR_HUNT_RADIUS_SQ);
    
    // Clamp the ratio between 0 and 1. If distance is greater than radius, it's 1.0.
    float distance_ratio = std::clamp(distance_to_threat / max_detection_radius, 0.0f, 1.0f);
    
    // A smaller distance should result in a lower bin index.
    // If distance_ratio is 1.0 (far away), it goes into the last bin.
    int distance_bin_index = static_cast<int>(distance_ratio * (NUM_DISTANCE_BINS - 0.0001f));
    distance_input_vec[distance_bin_index] = 1.0;
    state_brain_input_vec.insert(state_brain_input_vec.end(), distance_input_vec.begin(), distance_input_vec.end());

    // 2. Get the one-hot target vector for the current "correct" state.
    size_t target_state_id = static_cast<size_t>(m_state);
    const auto& target_state_vec = state_tokenizer.find_vector(target_state_id);

    // 3. Perform a training step.
    // Since this is a single-step prediction, not a sequence, we reset the state each time.
    m_state_brain.reset_state();
    
    // The inputs and outputs for backprop must be sequences (vector of vectors).
    std::vector<std::vector<double>> inputs_sequence = { state_brain_input_vec };
    
    // Forward pass to get the prediction
    std::vector<double> predicted_output = m_state_brain.forward_pass(state_brain_input_vec);
    std::vector<std::vector<double>> outputs_sequence = { predicted_output };
	
    // Expected output sequence
    std::vector<std::vector<double>> expected_sequence = { target_state_vec };

    // Backpropagate to learn
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

