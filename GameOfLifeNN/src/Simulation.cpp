#include "Simulation.h"
#include <glm/gtx/norm.hpp>
#include <random>
#include <chrono>

// Use a random number generator for initial placement and mutations
static auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
static std::mt19937 gen(seed);
static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
static std::uniform_real_distribution<float> pos_dis(0.0f, 1.0f); // New distribution for positions

Simulation::Simulation(size_t initial_agents, size_t initial_food, float world_size)
    : m_world_size(world_size)
{
    // Initialize with a few agents and some food
    spawn_initial_agents(initial_agents);
    spawn_new_food(initial_food, 0.8f);
}

void Simulation::tick() {
    // Lock the data to prevent race conditions with rendering thread (if any)
    std::lock_guard<std::mutex> lock(m_data_mutex);

    // 1. Update each agent
    for (auto& agent : m_agents) {
        agent.update(*this);
    }

    // 2. Handle interactions (eating, etc.)
    handle_agent_interactions();

    // 3. Handle death and reproduction
    cleanup_dead_agents_and_spawn_offspring();
    
	/* Auto Feeder.
    // If agent population is low, add more food to help them recover.
    if(m_agents.size() < 20 && m_food.size() < 20)
	{
        // FIX: Call the non-locking spawn_food since m_data_mutex is already locked.
        spawn_food(100, 0.1f);
    }
	*/
}

void Simulation::handle_agent_interactions() {
    // Iterate through agents and food to check for eating
    for (auto& agent : m_agents) {
        // Find the index of the nearest food
        int nearest_food_idx = -1;
        float min_dist_sq = 1e9;
        glm::vec3 agent_pos = glm::vec3(agent.m_transform[3]);

        for (size_t i = 0; i < m_food.size(); ++i) {
            float dist_sq = glm::distance2(agent_pos, m_food[i].position);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_food_idx = i;
            }
        }

        // Check if the nearest food is within eating distance
        if (nearest_food_idx != -1 && min_dist_sq < MIN_DIST_TO_EAT_SQ)
		{
            agent.m_energy += 50.0f; // Give a burst of energy
            m_food.erase(m_food.begin() + nearest_food_idx); // Remove the eaten food
        }
    }
}

void Simulation::cleanup_dead_agents_and_spawn_offspring()
{
    // A temporary vector to hold new agents
    std::vector<Agent> new_agents;
	
    // A temporary vector to hold agents that are still alive and healthy
    std::vector<Agent> surviving_agents;
	
    for(auto& agent : m_agents)
	{
        if(agent.isDead())
		{
			// Agent is dead. Spawn food.
            // FIX: Call the non-locking spawn_food since m_data_mutex is already locked.
			spawn_food(1, 0.8f);
        } else if(agent.canReproduce()) {
            // Agent can reproduce, create a child and keep the parent
            new_agents.push_back(agent.reproduce());
            surviving_agents.push_back(std::move(agent));
        } else
		{
            // Agent is alive but cannot reproduce
            surviving_agents.push_back(std::move(agent));
        }
    }
	
    // Replace the old agent list with the surviving and new agents
    m_agents = std::move(surviving_agents);
    m_agents.insert(m_agents.end(), std::make_move_iterator(new_agents.begin()), std::make_move_iterator(new_agents.end()));
}


void Simulation::spawn_food(size_t count, float food_area_scale)
{
	// Hard limit on food.
	if(m_food.size() > 1000) return;
	
    const float food_spawn_radius = (m_world_size * food_area_scale) / 2.0f;
	
    for(size_t i = 0; i < count; ++i)
	{
        // 'dis' is a uniform distribution from -1.0 to 1.0.
        // Multiplying by the radius gives a random point within the centered cube.
        glm::vec3 pos(dis(gen) * food_spawn_radius,
                      dis(gen) * food_spawn_radius,
                      dis(gen) * food_spawn_radius);
        m_food.push_back({pos});
    }
}

// Public method to spawn food at runtime. It's thread-safe.
void Simulation::spawn_new_food(size_t count, float food_area_scale)
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    spawn_food(count, food_area_scale);
}


// Public method to spawn agents at runtime. It's thread-safe.
void Simulation::spawn_new_agents(size_t count)
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    spawn_agents_internal(count);
}

// This function now calls the internal helper. No lock needed as it's called from constructor.
void Simulation::spawn_initial_agents(size_t count) {
    spawn_agents_internal(count);
}

// Internal helper function with the actual spawning logic to avoid code duplication.
void Simulation::spawn_agents_internal(size_t count) {
    float half_world_size = m_world_size / 2.0f;
    for(size_t i = 0; i < count; ++i)
	{
        // The perception vector has 7 inputs.
        // The brain must be initialized to accept all sensory inputs.
        constexpr size_t perception_vector_size = 7;
        ls3DRNN brain(perception_vector_size, {32, 32, 32});
        
        // Initial position in a random location within the world
        glm::vec3 initial_pos(pos_dis(gen) * m_world_size - half_world_size,
			pos_dis(gen) * m_world_size - half_world_size,
			pos_dis(gen) * m_world_size - half_world_size);
		
        glm::mat4 initial_transform = glm::translate(glm::mat4(1.0f), initial_pos);
		
        m_agents.emplace_back(brain, initial_transform);
    }
}
