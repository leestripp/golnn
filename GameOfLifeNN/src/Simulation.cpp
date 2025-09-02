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
	: m_world_size(world_size)
{
	// Initialize with a few agents and some food
	spawn_initial_agents(initial_agents);
	spawn_new_food(initial_food, 0.8f); // This will spawn VEGETABLE food by default
}

void Simulation::tick() {
	// Lock the data to prevent race conditions with rendering thread (if any)
	std::lock_guard<std::mutex> lock(m_data_mutex);

	// 1. Update each agent
	for (auto& agent : m_agents) {
		agent.update(*this);
	}

	// 2. Handle interactions (eating, predation)
	handle_agent_interactions();

	// 3. Handle death and reproduction
	cleanup_dead_agents_and_spawn_offspring();
}

void Simulation::handle_agent_interactions()
{
	// --- Phase 1: Agents eat food ---
	std::vector<size_t> eaten_food_indices;
	for(auto& agent : m_agents)
	{
		if (agent.isDead()) continue;

		glm::vec3 agent_pos = glm::vec3(agent.m_transform[3]);
		for(size_t i = 0; i < m_food.size(); ++i)
		{
			// Check if food has already been marked for removal
			if (std::find(eaten_food_indices.begin(), eaten_food_indices.end(), i) != eaten_food_indices.end()) {
				continue;
			}

			// Check if agent can eat this food type
			bool can_eat = (agent.m_type == Agent::AgentType::FEEDER && m_food[i].m_food_type == FoodType::VEGETABLE) ||
			               (agent.m_type == Agent::AgentType::PREDATOR && m_food[i].m_food_type == FoodType::MEAT);

			if (can_eat && glm::distance2(agent_pos, m_food[i].position) < MIN_DIST_TO_EAT_SQ) {
				agent.m_energy += 50.0f;
				eaten_food_indices.push_back(i);
				break; // This agent has eaten, move to the next agent
			}
		}
	}
	
	// Remove all eaten food at once. Sort indices descending to prevent invalidation.
	std::sort(eaten_food_indices.rbegin(), eaten_food_indices.rend());
	for (size_t index : eaten_food_indices) {
		m_food.erase(m_food.begin() + index);
	}

	// --- Phase 2: Predators hunt ---
	for (auto& predator : m_agents) {
		if (predator.m_type != Agent::AgentType::PREDATOR || predator.isDead()) continue;

		glm::vec3 predator_pos = glm::vec3(predator.m_transform[3]);
		Agent* closest_feeder_prey = nullptr;
		float min_feeder_dist_sq = PREDATION_DISTANCE_SQ;

		// First, prioritize finding the closest feeder in attack range
		for (auto& potential_prey : m_agents) {
			if (potential_prey.m_type != Agent::AgentType::FEEDER || potential_prey.isDead()) continue;
			
			float dist_sq = glm::distance2(predator_pos, glm::vec3(potential_prey.m_transform[3]));
			if (dist_sq < min_feeder_dist_sq) {
				min_feeder_dist_sq = dist_sq;
				closest_feeder_prey = &potential_prey;
			}
		}

		if (closest_feeder_prey) {
			// An attack occurs on a feeder! Energy is transferred, not created.
			predator.m_energy += PREDATION_ENERGY_TRANSFER;
			closest_feeder_prey->m_energy -= PREDATION_ENERGY_TRANSFER;
			continue; // This predator has acted, move to the next one
		}

		// If no feeder was in range to be attacked, look for other predators
		Agent* closest_predator_prey = nullptr;
		float min_predator_dist_sq = PREDATION_DISTANCE_SQ;

		for (auto& potential_prey : m_agents) {
			// Can't attack itself, and must be a predator
			if (&potential_prey == &predator || potential_prey.m_type != Agent::AgentType::PREDATOR || potential_prey.isDead()) continue;
			
			float dist_sq = glm::distance2(predator_pos, glm::vec3(potential_prey.m_transform[3]));
			if (dist_sq < min_predator_dist_sq) {
				min_predator_dist_sq = dist_sq;
				closest_predator_prey = &potential_prey;
			}
		}

		if (closest_predator_prey) {
			// An attack occurs on another predator! Energy is transferred.
			predator.m_energy += PREDATION_ENERGY_TRANSFER;
			closest_predator_prey->m_energy -= PREDATION_ENERGY_TRANSFER;
		}
	}
}

void Simulation::cleanup_dead_agents_and_spawn_offspring()
{
	// Check for the "last agent dying" scenario BEFORE processing
	if(m_agents.size() == 1 && m_agents[0].isDead())
	{
		const Agent& last_agent = m_agents[0];
		std::vector<size_t> hidden_dims = last_agent.m_brain.get_hidden_layer_dims();
		std::string agent_type_str = (last_agent.m_type == Agent::AgentType::PREDATOR) ? "Predator" : "Feeder";
			
	// Output: Agent type, layer count and neuron count per layer.
		std::cout << "\n--- Alpha Agent Died ---\n";
		std::cout << "Type: " << agent_type_str << "\n";
		std::cout << "Hidden Layers: " << hidden_dims.size() << "\n";
		std::cout << "Neurons per Layer: [ ";
		for (size_t i = 0; i < hidden_dims.size(); ++i) {
			std::cout << hidden_dims[i] << (i == hidden_dims.size() - 1 ? "" : ", ");
		}
		std::cout << " ]\n--------------------------\n" << std::endl;
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
				m_food.push_back({FoodType::MEAT, glm::vec3(agent.m_transform[3])});
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

		// The perception vector has 17 inputs:
		// - Food: World position (x, y, z) & distance (1) -> 4
		// - Follow Agent: World position (x, y, z) & distance (1) -> 4
		// - Threat/Prey: World position (x, y, z) & distance (1) -> 4
		// - Own energy (1)
		// - Current State (one-hot encoded) (4)
		constexpr size_t perception_vector_size = 17;

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
