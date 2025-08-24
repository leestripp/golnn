
#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <glm/glm.hpp>
#include <mutex>

#include "Agent.h"

// A simple struct to represent food in the world
struct Food {
    glm::vec3 position;
};

class Simulation {
public:
    Simulation(size_t initial_agents, size_t initial_food, float world_size);

    // The main function that runs one step of the simulation
    void tick();

    // Spawns a specified number of new agents during runtime
    void spawn_new_agents(size_t count);
	
    // Spawns a specified number of new food items during runtime
    void spawn_new_food(size_t count, float food_area_scale = 0.5f);
	
    // -- Public State --
    // Agents and food are public so the main loop can access them for rendering
    std::vector<Agent> m_agents;
    std::vector<Food> m_food;
    const float m_world_size;
    std::mutex m_data_mutex; // Mutex to protect agent/food vectors during access
	
private:
    // Helper functions for the simulation loop
    void handle_agent_interactions();
    void cleanup_dead_agents_and_spawn_offspring();
    void spawn_food(size_t count, float food_area_scale);
    void spawn_agents_internal(size_t count);
    void spawn_initial_agents(size_t count);
	
    // Static variables for simulation parameters
    static constexpr float MIN_DIST_TO_EAT_SQ = 10.0f; // 5m = 25sd Squared distance to eat food
};

#endif // SIMULATION_H
