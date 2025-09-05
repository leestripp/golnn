#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <glm/glm.hpp>
#include <mutex>

#include "Agent.h"

// Define food types
enum FoodType : size_t {
    VEGETABLE = 0,
    MEAT = 1
};

// A simple struct to represent food in the world
struct Food {
	size_t m_food_type;
    glm::vec3 position;
};

// A simple struct to represent a sound event in the world for one tick
struct SoundEvent {
    glm::vec3 position;
    float intensity;
};

class Simulation
{
public:
    Simulation(size_t initial_agents, size_t initial_food, float world_size);
	
    // The main function that runs one step of the simulation
    void tick();

    // Spawns a specified number of new agents during runtime
    void spawn_new_agents(size_t count);
    // Spawns a specified number of new food items during runtime
    void spawn_new_food(size_t count, float food_area_scale = 0.8f);
	
    // Public State
    // Agents and food are public so the main loop can access them for rendering
    std::vector<Agent> m_agents;
    std::vector<Food> m_food;
    std::vector<SoundEvent> m_sound_events; // Holds sound events for the current tick
    const float m_world_size;
    std::mutex m_data_mutex; // Mutex to protect agent/food vectors during access
	
    // --- Simulation Parameters ---
	// Brain structure parameters for new agents
	static constexpr int MAX_LAYER_COUNT = 10;
	static constexpr int MAX_NEURON_COUNT = 64;
	
private:
    // Helper functions for the simulation loop
    void handle_agent_interactions();
    void cleanup_dead_agents_and_spawn_offspring();
    void spawn_food(size_t count, float food_area_scale, FoodType type = FoodType::VEGETABLE);
    void spawn_agents_internal(size_t count);
    void spawn_initial_agents(size_t count);
	
    size_t m_food_spawn_timer; // Timer for food regrowth
	
	// S	tatic variables for simulation parameters
	static constexpr float MIN_DIST_TO_EAT_SQ = 1.0f;			// Squared distance to eat food (1 unit)
	static constexpr float PREDATION_DISTANCE_SQ = 1.0f;			// Squared distance for a predator to attack (1 unit)
	// Predation is now a direct energy transfer. The drain from prey equals the gain for the predator.
	static constexpr float PREDATION_ENERGY_TRANSFER = 5.0f;	// Energy transferred from prey to predator
	// Limits
	static constexpr size_t MAX_AGENTS = 1000;
	static constexpr size_t MAX_FOOD = 1000;
    // Food regrowth parameters
    static constexpr size_t FOOD_SPAWN_INTERVAL_TICKS = 500; // Spawn food every N ticks
    static constexpr size_t FOOD_SPAWN_AMOUNT = 5;          // Number of food items to spawn
};

#endif // SIMULATION_H
