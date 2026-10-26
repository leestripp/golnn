#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <glm/glm.hpp>
#include <mutex>
#include <memory>
#include <atomic>
#include <array>
#include <iosfwd>
#include <random>

#include "SimulationConfig.h"
#include "StateChecksum.h"
#include "Agent.h"
#include "RenderSnapshot.h"

class ThreadPool;

#include "SpatialGrid.h"

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
    Simulation(size_t initial_agents, size_t initial_food, float world_size, SimulationConfig config = {});
	~Simulation();
	
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
    std::vector<SoundEvent> m_sound_events; // New events from this tick's interactions
    std::vector<SoundEvent> m_audible_sound_events; // Previous tick, sensed now
    const float m_world_size;
    std::mutex m_data_mutex; // Mutex to protect agent/food vectors during access

    // Spatial indices over agents and food, partitioned by type and rebuilt
    // each tick. Agents query these instead of linearly scanning m_agents /
    // m_food, and each sensor only visits the grid holding the types it can
    // actually detect. Cell size matches the largest local sensor radius
    // (scent/sound = 30 units).
    SpatialGrid m_feeder_grid;
    SpatialGrid m_predator_grid;
    SpatialGrid m_vegetable_grid;
    SpatialGrid m_meat_grid;
    static constexpr float SPATIAL_CELL_SIZE = 30.0f;
    void rebuild_spatial_grids();
	state_checksum::type state_checksum() const;
	const SimulationConfig& get_config() const { return m_config; }
	std::size_t worker_count() const;
	EcologyStats ecology_stats() const;
	std::shared_ptr<const WorldSnapshot> render_snapshot() const
	{
		return m_render_snapshot.load();
	}
	void set_brain_tier(BrainTier tier);
	const glm::mat4& sensed_agent_transform(std::size_t index) const
	{
		return m_agent_transform_snapshot[index];
	}
	
private:
    // Helper functions for the simulation loop
	void update_food_regrowth();
	void train_movement_brains();
	void write_telemetry();
	void publish_render_snapshot();
	void rebuild_agent_grids();
	void rebuild_food_grids();
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
	SimulationConfig m_config;
	std::mt19937 m_rng;
	std::unique_ptr<ThreadPool> m_thread_pool;

	// Frozen before the parallel update phase. Agents sense other agents from
	// this snapshot, never from transforms being mutated by other workers.
	std::vector<glm::mat4> m_agent_transform_snapshot;

	// Reused interaction-phase scratch; capacities persist across ticks.
	std::vector<bool> m_food_eaten_scratch;
	std::vector<Agent*> m_predator_scratch;
	std::vector<Agent*> m_feeder_scratch;

	struct MovementBatch
	{
		std::vector<Agent*> agents;
		std::vector<ls3DRNN*> brains;
		std::vector<const std::vector<float>*> commands;
		std::vector<const std::vector<float>*> targets;
	};
	std::array<MovementBatch, BRAIN_TIER_COUNT> m_movement_batches;
	std::vector<Agent> m_offspring_scratch;

	std::size_t m_tick_count = 0;
	std::size_t m_total_births = 0;
	std::size_t m_total_deaths = 0;
	std::size_t m_total_vegetables_eaten = 0;
	std::size_t m_vegetables_eaten_since_balance = 0;
	std::unique_ptr<std::ofstream> m_telemetry_output;

	// Two reusable slots avoid steady-state render allocation. If a renderer
	// still holds the next slot, allocate a replacement instead of mutating it.
	std::array<std::shared_ptr<WorldSnapshot>, 2> m_render_snapshot_storage{
		std::make_shared<WorldSnapshot>(), std::make_shared<WorldSnapshot>()};
	std::size_t m_render_write_index = 0;
	std::atomic<std::shared_ptr<const WorldSnapshot>> m_render_snapshot;
};

#endif // SIMULATION_H
