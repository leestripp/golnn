#include "Simulation.h"
#include <glm/gtx/norm.hpp>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <string>

// Use a random number generator for initial placement and mutations
static auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
static std::mt19937 gen(seed);
static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
static std::uniform_real_distribution<float> pos_dis(0.0f, 1.0f);
static std::uniform_real_distribution<float> chance_dis(0.0f, 1.0f); // For random chances

// Brain structure randomization
// The distributions are initialized using the constants defined in Simulation.h
static std::uniform_int_distribution<int> layer_dist(1, Simulation::MAX_LAYER_COUNT);
static std::uniform_int_distribution<int> neuron_dist(4, Simulation::MAX_NEURON_COUNT);


Simulation::Simulation(size_t initial_agents, size_t initial_food, float world_size)
	: m_world_size(world_size),
	m_feeder_grid(world_size, SPATIAL_CELL_SIZE),
	m_predator_grid(world_size, SPATIAL_CELL_SIZE),
	m_vegetable_grid(world_size, SPATIAL_CELL_SIZE),
	m_meat_grid(world_size, SPATIAL_CELL_SIZE),
	m_food_spawn_timer(0)
{
	// Initialize with a few agents and some food
	spawn_initial_agents(initial_agents);
	spawn_new_food(initial_food, 0.8f); // This will spawn VEGETABLE food by default
}


void Simulation::tick() {
	// Lock the data to prevent race conditions with rendering thread
	std::lock_guard<std::mutex> lock(m_data_mutex);

    // Food Regrowth Timer
    m_food_spawn_timer++;
    if(m_food_spawn_timer >= FOOD_SPAWN_INTERVAL_TICKS)
    {
        spawn_food(FOOD_SPAWN_AMOUNT, 0.8f, FoodType::VEGETABLE);
        m_food_spawn_timer = 0;
    }
	
	// Rebuild the spatial indices for this tick (O(n)). All proximity queries
	// in perceive()/handle_agent_interactions() hit these instead of O(n) scans.
	rebuild_spatial_grids();

	// 1. Update each agent
	for(auto& agent : m_agents)
	{
		agent.update(*this);
	}

	// Agents moved during update: refresh the agent grid for interactions.
	rebuild_spatial_grids();

	// 2. Handle interactions (eating, predation)
	handle_agent_interactions();
	
	// 3. Handle death and reproduction
	cleanup_dead_agents_and_spawn_offspring();

	// 4. Clear transient world data for the next tick
	m_sound_events.clear();
}


void Simulation::rebuild_spatial_grids()
{
	m_feeder_grid.clear();
	m_predator_grid.clear();
	for (size_t i = 0; i < m_agents.size(); ++i)
	{
		// Dead agents are skipped by the query sites, but keeping them out of
		// the grid entirely is cheaper.
		if (m_agents[i].isDead()) continue;
		if (m_agents[i].m_type == Agent::AgentType::FEEDER)
			m_feeder_grid.insert(glm::vec3(m_agents[i].m_transform[3]), i);
		else
			m_predator_grid.insert(glm::vec3(m_agents[i].m_transform[3]), i);
	}

	m_vegetable_grid.clear();
	m_meat_grid.clear();
	for (size_t i = 0; i < m_food.size(); ++i)
	{
		if (m_food[i].m_food_type == FoodType::VEGETABLE)
			m_vegetable_grid.insert(m_food[i].position, i);
		else
			m_meat_grid.insert(m_food[i].position, i);
	}
}


void Simulation::handle_agent_interactions()
{
	// Use a boolean vector for efficient O(1) lookup of eaten food.
	std::vector<bool> food_eaten(m_food.size(), false);
	std::vector<Agent*> predators;
	std::vector<Agent*> feeders;

	// Separate agents by type for cleaner and more efficient processing.
	for (auto& agent : m_agents) {
		if (agent.isDead()) continue;
		if (agent.m_type == Agent::AgentType::PREDATOR) {
			predators.push_back(&agent);
		} else {
			feeders.push_back(&agent);
		}
	}

	// --- Phase 1: Feeders eat VEGETABLE food ---
	// This is handled first as it's a simple, non-conflicting action.
	// Eating radius is 1 unit, far below the grid cell size, so each feeder
	// only inspects its own cell (plus a boundary ring).
	const float eat_radius = sqrt(MIN_DIST_TO_EAT_SQ);
	for (auto* feeder : feeders) {
		glm::vec3 agent_pos = glm::vec3(feeder->m_transform[3]);
		m_vegetable_grid.visit(agent_pos, eat_radius, [&](size_t i) {
			if (food_eaten[i]) return;

			if (glm::distance2(agent_pos, m_food[i].position) < MIN_DIST_TO_EAT_SQ) {
				feeder->m_energy += 50.0f;
				food_eaten[i] = true;
				m_sound_events.push_back({m_food[i].position, 0.5f});
			}
		});
	}

	// --- Phase 2: Predators act based on a strict priority list ---
	// For each predator, we check for prey in a specific order. If an action is taken,
	// we move to the next predator. This correctly models the priority.
	for (auto* predator : predators) {
		glm::vec3 predator_pos = glm::vec3(predator->m_transform[3]);
		bool action_taken = false;

		// Priority 1: Hunt Feeders
		Agent* closest_feeder_prey = nullptr;
		float min_feeder_dist_sq = PREDATION_DISTANCE_SQ;
		m_feeder_grid.visit(predator_pos, sqrt(PREDATION_DISTANCE_SQ), [&](size_t i) {
			Agent* potential_prey = &m_agents[i];
			if (potential_prey->isDead()) return;

			float dist_sq = glm::distance2(predator_pos, glm::vec3(potential_prey->m_transform[3]));
			if (dist_sq < min_feeder_dist_sq) {
				min_feeder_dist_sq = dist_sq;
				closest_feeder_prey = potential_prey;
			}
		});

		if (closest_feeder_prey) {
			predator->m_energy += PREDATION_ENERGY_TRANSFER;
			closest_feeder_prey->m_energy -= PREDATION_ENERGY_TRANSFER;
			m_sound_events.push_back({glm::vec3(closest_feeder_prey->m_transform[3]), 1.0f});
			action_taken = true;
		}

		if (action_taken) continue; // Predator has acted, move to the next one.

		// Priority 2: Eat Meat
		int closest_meat_index = -1;
		float min_meat_dist_sq = MIN_DIST_TO_EAT_SQ;
		m_meat_grid.visit(predator_pos, eat_radius, [&](size_t i) {
			if (food_eaten[i]) return;

			float dist_sq = glm::distance2(predator_pos, m_food[i].position);
			if (dist_sq < min_meat_dist_sq) {
				min_meat_dist_sq = dist_sq;
				closest_meat_index = static_cast<int>(i);
			}
		});

		if (closest_meat_index != -1) {
			predator->m_energy += 50.0f;
			food_eaten[closest_meat_index] = true;
			m_sound_events.push_back({m_food[closest_meat_index].position, 0.5f});
			action_taken = true;
		}

		if (action_taken) continue; // Predator has acted, move to the next one.

		// Priority 3: Hunt other Predators (Cannibalism)
		Agent* closest_predator_prey = nullptr;
		float min_predator_dist_sq = PREDATION_DISTANCE_SQ;
		m_predator_grid.visit(predator_pos, sqrt(PREDATION_DISTANCE_SQ), [&](size_t i) {
			Agent* potential_prey = &m_agents[i];
			if (potential_prey == predator || potential_prey->isDead()) return;

			float dist_sq = glm::distance2(predator_pos, glm::vec3(potential_prey->m_transform[3]));
			if (dist_sq < min_predator_dist_sq) {
				min_predator_dist_sq = dist_sq;
				closest_predator_prey = potential_prey;
			}
		});

		if (closest_predator_prey) {
			predator->m_energy += PREDATION_ENERGY_TRANSFER;
			closest_predator_prey->m_energy -= PREDATION_ENERGY_TRANSFER;
			m_sound_events.push_back({glm::vec3(closest_predator_prey->m_transform[3]), 1.0f});
		}
	}

	// --- Phase 3: Remove all eaten food from the world ---
	// This approach is safer and cleaner than erasing elements while iterating.
	std::vector<Food> remaining_food;
	remaining_food.reserve(m_food.size());
	for (size_t i = 0; i < m_food.size(); ++i) {
		if (!food_eaten[i]) {
			remaining_food.push_back(m_food[i]);
		}
	}
	m_food = std::move(remaining_food);
}


void Simulation::cleanup_dead_agents_and_spawn_offspring()
{
	// Check for the "last agent dying" scenario BEFORE processing
	if(m_agents.size() == 1 && m_agents[0].isDead())
	{
		const Agent& last_agent = m_agents[0];
		std::string agent_type_str = (last_agent.m_type == Agent::AgentType::PREDATOR) ? "Predator" : "Feeder";
		
		std::vector<size_t> ls3d_hidden_dims = last_agent.m_brain.get_hidden_layer_dims();
		size_t ls_hidden_dim = last_agent.get_state_brain_hidden_dim();
		float learning_rate = last_agent.get_learning_rate();
		
		std::cout << "\n--- Alpha Agent Died ---\n";
		std::cout << "Type: " << agent_type_str << "\n";
		std::cout << "Learning Rate: " << learning_rate << "\n";
		
		std::cout << "\n-- Movement Brain (ls3DRNN) --\n";
		std::cout << "Hidden Layers: " << ls3d_hidden_dims.size() << "\n";
		std::cout << "Neurons per Layer: [ ";
		if (ls3d_hidden_dims.empty()) {
			std::cout << "N/A";
		} else {
			for (size_t i = 0; i < ls3d_hidden_dims.size(); ++i) {
				std::cout << ls3d_hidden_dims[i] << (i == ls3d_hidden_dims.size() - 1 ? "" : ", ");
			}
		}
		std::cout << " ]\n";
		
		std::cout << "\n-- State Brain (lsRNN) --\n";
		std::cout << "Hidden Layers: 1\n";
		std::cout << "Neurons per Layer: [ " << ls_hidden_dim << " ]\n";
		
		std::cout << "--------------------------\n" << std::endl;
		
		// Clear all food.
		m_food.clear();
	}
	
	std::vector<Agent> surviving_agents;
	surviving_agents.reserve(m_agents.size()); // Optimization: avoid reallocations for survivors
	std::vector<Agent> new_agents;
	
	for(auto& agent : m_agents)
	{
		if(agent.isDead())
		{
			// When a feeder dies, it leaves meat behind for predators.
			if(m_food.size() < MAX_FOOD && agent.m_type == Agent::AgentType::FEEDER)
			{
				glm::vec3 food_pos = glm::vec3(agent.m_transform[3]);
				float half_world_size = m_world_size / 2.0f;

				// Ensure the food spawns within the world boundaries.
				food_pos.x = std::clamp(food_pos.x, -half_world_size, half_world_size);
				food_pos.y = std::clamp(food_pos.y, -half_world_size, half_world_size);
				food_pos.z = std::clamp(food_pos.z, -half_world_size, half_world_size);
				
				m_food.push_back({FoodType::MEAT, food_pos});
			}
		} else // Agent is alive
		{
			// Check if the agent can reproduce AND if there is room for offspring.
			// This check is crucial to prevent the population from exploding and crashing.
			if(agent.canReproduce() && (m_agents.size() + new_agents.size() < MAX_AGENTS))
			{
				new_agents.push_back(agent.reproduce());
			}
			// The parent always survives this tick.
			surviving_agents.push_back(std::move(agent));
		}
	}
	
	// Replace the old agent list with the surviving agents
	m_agents = std::move(surviving_agents);
	// Add the new offspring to the list
	if(! new_agents.empty())
	{
		m_agents.insert(m_agents.end(), std::make_move_iterator(new_agents.begin()), std::make_move_iterator(new_agents.end()));
	}
}


void Simulation::spawn_food(size_t count, float food_area_scale, FoodType type)
{
	// Hard limit on food.
	if(m_food.size() >= MAX_FOOD)
	{
		return;
	}
	
	const float food_spawn_radius = (m_world_size * food_area_scale) / 2.0f;
	
	for(size_t i = 0; i < count; ++i)
	{
		// Always check our food count.
		if(m_food.size() >= MAX_FOOD) break;
		
		// 'dis' is a uniform distribution from -1.0 to 1.0.
		// Multiplying by the radius gives a random point within the centered cube.
		glm::vec3 pos(dis(gen) * food_spawn_radius,
					  dis(gen) * food_spawn_radius,
					  dis(gen) * food_spawn_radius);
		m_food.push_back({type, pos});
	}
}

// Public method to spawn food at runtime. It's thread-safe.
void Simulation::spawn_new_food(size_t count, float food_area_scale)
{
	std::lock_guard<std::mutex> lock(m_data_mutex);
	spawn_food(count, food_area_scale, FoodType::VEGETABLE);
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
void Simulation::spawn_agents_internal(size_t count)
{
	float half_world_size = m_world_size / 2.0f;
	for(size_t i = 0; i < count; ++i)
	{
		// Stop spawning if we've hit the agent limit.
		if(m_agents.size() >= MAX_AGENTS)
		{
			break;
		}

		// The perception vector size is now defined in the Agent class.
		constexpr size_t perception_vector_size = Agent::PERCEPTION_VECTOR_SIZE;

		// Create a brain with a random number of layers and neurons.
		int num_hidden_layers = layer_dist(gen);
		std::vector<size_t> hidden_dims;
		hidden_dims.reserve(num_hidden_layers);
		for(int j = 0; j < num_hidden_layers; ++j)
		{
			hidden_dims.push_back(neuron_dist(gen));
		}
		ls3DRNN brain(perception_vector_size, hidden_dims);
		
		// Initial position in a random location within the world
		glm::vec3 initial_pos(pos_dis(gen) * m_world_size - half_world_size,
			pos_dis(gen) * m_world_size - half_world_size,
			pos_dis(gen) * m_world_size - half_world_size);
		
		glm::mat4 initial_transform = glm::translate(glm::mat4(1.0f), initial_pos);
		
		// Determine agent type: 10% chance to be a predator
		Agent::AgentType type = (chance_dis(gen) < 0.1f) ? Agent::AgentType::PREDATOR : Agent::AgentType::FEEDER;

		m_agents.emplace_back(brain, initial_transform, type);
	}
}
