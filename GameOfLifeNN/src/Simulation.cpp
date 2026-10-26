#include "Simulation.h"
#include "ThreadPool.h"
#include <glm/gtx/norm.hpp>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cmath>

Simulation::Simulation(size_t initial_agents, size_t initial_food, float world_size, SimulationConfig config)
	: m_world_size(world_size),
	m_feeder_grid(world_size, SPATIAL_CELL_SIZE),
	m_predator_grid(world_size, SPATIAL_CELL_SIZE),
	m_vegetable_grid(world_size, SPATIAL_CELL_SIZE),
	m_meat_grid(world_size, SPATIAL_CELL_SIZE),
	m_food_spawn_timer(0),
	m_config(std::move(config)),
	m_rng(m_config.seed)
{
	const std::size_t hardware_workers = std::max(1u, std::thread::hardware_concurrency());
	const std::size_t workers = m_config.worker_count == 0 ? hardware_workers : m_config.worker_count;
	m_thread_pool = std::make_unique<ThreadPool>(workers);
	m_agents.reserve(m_config.max_agents);
	m_food.reserve(m_config.max_food);
	if(!m_config.telemetry_path.empty())
	{
		m_telemetry_output = std::make_unique<std::ofstream>(m_config.telemetry_path);
		if(!*m_telemetry_output)
			throw std::runtime_error("Unable to open telemetry file: " + m_config.telemetry_path);
		*m_telemetry_output << "tick,feeders,predators,vegetables,meat,births,deaths,"
		                       "vegetables_eaten,mean_energy,mean_age,idle,roam,hunt,flee,"
		                       "brain_small,brain_medium,brain_large\n";
	}
	// Initialize with a few agents and some food
	spawn_initial_agents(initial_agents);
	spawn_new_food(initial_food, 0.8f); // This will spawn VEGETABLE food by default
}


void Simulation::tick() {
	// Lock the data to prevent race conditions with rendering thread
	std::lock_guard<std::mutex> lock(m_data_mutex);

	update_food_regrowth();
	
	// Rebuild the spatial indices for this tick (O(n)). All proximity queries
	// in perceive()/handle_agent_interactions() hit these instead of O(n) scans.
	rebuild_spatial_grids();

	// 1. Update independent agents against the frozen start-of-tick snapshot.
	// Interactions and reproduction remain serial and in a stable order below.
	const std::size_t agent_count = m_agents.size();
	const std::size_t grain_size = std::max<std::size_t>(1, agent_count / (worker_count() * 4));
	m_thread_pool->parallel_for(agent_count, grain_size, [&](const std::size_t begin, const std::size_t end) {
		for(std::size_t i = begin; i < end; ++i) m_agents[i].prepare_update(*this);
	});
	train_movement_brains();
	m_thread_pool->parallel_for(agent_count, grain_size, [&](const std::size_t begin, const std::size_t end) {
		for(std::size_t i = begin; i < end; ++i) m_agents[i].finish_update(*this);
	});

	// Agents moved during update; food stayed put. Refresh only agent grids.
	rebuild_agent_grids();

	// 2. Handle interactions (eating, predation)
	handle_agent_interactions();
	
	// 3. Handle death and reproduction
	cleanup_dead_agents_and_spawn_offspring();

	// 4. Clear transient world data and publish ecology telemetry.
	m_sound_events.clear();
	++m_tick_count;
	write_telemetry();
}


void Simulation::rebuild_spatial_grids()
{
	m_agent_transform_snapshot.resize(m_agents.size());
	for(std::size_t i = 0; i < m_agents.size(); ++i)
		m_agent_transform_snapshot[i] = m_agents[i].m_transform;
	rebuild_agent_grids();
	rebuild_food_grids();
}

void Simulation::rebuild_agent_grids()
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
}

void Simulation::rebuild_food_grids()
{
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
	m_food_eaten_scratch.assign(m_food.size(), false);
	std::vector<bool>& food_eaten = m_food_eaten_scratch;
	std::vector<Agent*>& predators = m_predator_scratch;
	std::vector<Agent*>& feeders = m_feeder_scratch;
	predators.clear();
	feeders.clear();

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
				++m_total_vegetables_eaten;
				++m_vegetables_eaten_since_balance;
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

	// --- Phase 3: Compact eaten food in place without a fresh vector each tick.
	size_t food_index = 0;
	std::erase_if(m_food, [&](const Food&) { return food_eaten[food_index++]; });
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
			++m_total_deaths;
			// When a feeder dies, it leaves meat behind for predators.
			if(m_food.size() < m_config.max_food && agent.m_type == Agent::AgentType::FEEDER)
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
			if(agent.canReproduce() && (m_agents.size() + new_agents.size() < m_config.max_agents))
			{
				new_agents.push_back(agent.reproduce());
			}
			// The parent always survives this tick.
			surviving_agents.push_back(std::move(agent));
		}
	}
	
	// Replace the old agent list with the surviving agents
	m_agents = std::move(surviving_agents);
	m_total_births += new_agents.size();
	// Add the new offspring to the list
	if(! new_agents.empty())
	{
		m_agents.insert(m_agents.end(), std::make_move_iterator(new_agents.begin()), std::make_move_iterator(new_agents.end()));
	}
}


void Simulation::spawn_food(size_t count, float food_area_scale, FoodType type)
{
	std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
	// Hard limit on food.
	if(m_food.size() >= m_config.max_food)
	{
		return;
	}
	
	const float food_spawn_radius = (m_world_size * food_area_scale) / 2.0f;
	
	for(size_t i = 0; i < count; ++i)
	{
		// Always check our food count.
		if(m_food.size() >= m_config.max_food) break;
		
		// 'dis' is a uniform distribution from -1.0 to 1.0.
		// Multiplying by the radius gives a random point within the centered cube.
		glm::vec3 pos(dis(m_rng) * food_spawn_radius,
					  dis(m_rng) * food_spawn_radius,
					  dis(m_rng) * food_spawn_radius);
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
	std::uniform_real_distribution<float> pos_dis(0.0f, 1.0f);
	std::uniform_real_distribution<float> chance_dis(0.0f, 1.0f);
	float half_world_size = m_world_size / 2.0f;
	for(size_t i = 0; i < count; ++i)
	{
		// Stop spawning if we've hit the agent limit.
		if(m_agents.size() >= m_config.max_agents)
		{
			break;
		}

		// The perception vector size is now defined in the Agent class.
		constexpr size_t perception_vector_size = Agent::PERCEPTION_VECTOR_SIZE;

		// Standardized tensor shapes allow same-tier agents to share batched
		// forward/backward passes. Diversity comes from weights and hypergenes.
		const auto& hidden_dims = movement_hidden_dims(m_config.brain_tier);
		ls3DRNN brain(perception_vector_size, hidden_dims, m_rng);
		
		// Initial position in a random location within the world
		glm::vec3 initial_pos(pos_dis(m_rng) * m_world_size - half_world_size,
			pos_dis(m_rng) * m_world_size - half_world_size,
			pos_dis(m_rng) * m_world_size - half_world_size);
		
		glm::mat4 initial_transform = glm::translate(glm::mat4(1.0f), initial_pos);
		
		// Determine agent type: 10% chance to be a predator
		Agent::AgentType type = (chance_dis(m_rng) < m_config.predator_spawn_probability) ? Agent::AgentType::PREDATOR : Agent::AgentType::FEEDER;

		m_agents.emplace_back(brain, initial_transform, type, -1.0f, -1.0f,
			state_hidden_dim(m_config.brain_tier), m_rng(), m_config.brain_tier);
	}
}

void Simulation::train_movement_brains()
{
	const std::size_t agent_count = m_agents.size();
	const std::size_t grain_size = std::max<std::size_t>(1, agent_count / (worker_count() * 4));
	if(!m_config.use_batched_brains)
	{
		m_thread_pool->parallel_for(agent_count, grain_size, [&](const std::size_t begin, const std::size_t end) {
			for(std::size_t i = begin; i < end; ++i) m_agents[i].train_movement();
		});
		return;
	}

	for(MovementBatch& batch : m_movement_batches)
	{
		batch.agents.clear();
		batch.brains.clear();
		batch.commands.clear();
		batch.targets.clear();
	}
	for(Agent& agent : m_agents)
	{
		if(!agent.movement_pending()) continue;
		const std::size_t tier = static_cast<std::size_t>(agent.brain_tier());
		MovementBatch& batch = m_movement_batches[tier];
		batch.agents.push_back(&agent);
		batch.brains.push_back(&agent.m_brain);
		batch.commands.push_back(&agent.m_perception);
		batch.targets.push_back(&agent.movement_target());
	}

	for(Agent* agent : m_movement_batches[static_cast<std::size_t>(BrainTier::CUSTOM)].agents)
		agent->train_movement();

	for(std::size_t tier = static_cast<std::size_t>(BrainTier::SMALL); tier < BRAIN_TIER_COUNT; ++tier)
	{
		MovementBatch& batch = m_movement_batches[tier];
		const std::size_t count = batch.brains.size();
		if(count == 0) continue;
		const std::size_t batch_grain = std::max<std::size_t>(1, count / (worker_count() * 4));
		m_thread_pool->parallel_for(count, batch_grain, [&](const std::size_t begin, const std::size_t end) {
			const std::size_t length = end - begin;
			ls3DRNN::train_batch(
				std::span<ls3DRNN* const>(batch.brains).subspan(begin, length),
				std::span<const std::vector<float>* const>(batch.commands).subspan(begin, length),
				std::span<const std::vector<float>* const>(batch.targets).subspan(begin, length),
				Agent::movement_rotation_weight());
		});
	}
}

void Simulation::update_food_regrowth()
{
	++m_food_spawn_timer;
	if(m_config.food_regrowth_mode == FoodRegrowthMode::FIXED)
	{
		if(m_food_spawn_timer >= m_config.food_spawn_interval_ticks)
		{
			spawn_food(m_config.food_spawn_amount, 0.8f, FoodType::VEGETABLE);
			m_food_spawn_timer = 0;
		}
		return;
	}

	const std::size_t feeder_count = static_cast<std::size_t>(std::count_if(
		m_agents.begin(), m_agents.end(), [](const Agent& agent) {
			return !agent.isDead() && agent.m_type == Agent::AgentType::FEEDER;
		}));
	const std::size_t vegetable_count = static_cast<std::size_t>(std::count_if(
		m_food.begin(), m_food.end(), [](const Food& food) {
			return food.m_food_type == FoodType::VEGETABLE;
		}));

	// A small emergency floor prevents a zero-food tick from becoming an
	// unrecoverable extinction spiral. All spawning still respects MAX_FOOD.
	if(feeder_count > 0 && vegetable_count == 0)
		spawn_food(m_config.emergency_food_spawn, 0.8f, FoodType::VEGETABLE);

	if(m_food_spawn_timer < m_config.food_balance_interval_ticks) return;
	m_food_spawn_timer = 0;

	const float requested_target = static_cast<float>(feeder_count) * m_config.target_vegetation_per_feeder;
	const std::size_t target = requested_target > 0.0f
		? static_cast<std::size_t>(std::ceil(requested_target)) : 0;
	const std::size_t deficit = target > vegetable_count ? target - vegetable_count : 0;
	const std::size_t requested = deficit + m_vegetables_eaten_since_balance;
	const std::size_t spawn_count = std::min(requested, m_config.max_food_per_balance);
	if(spawn_count > 0) spawn_food(spawn_count, 0.8f, FoodType::VEGETABLE);
	m_vegetables_eaten_since_balance = 0;
}

EcologyStats Simulation::ecology_stats() const
{
	EcologyStats stats;
	stats.tick = m_tick_count;
	stats.total_births = m_total_births;
	stats.total_deaths = m_total_deaths;
	stats.total_vegetables_eaten = m_total_vegetables_eaten;
	double energy_sum = 0.0;
	for(const Agent& agent : m_agents)
	{
		if(agent.m_type == Agent::AgentType::FEEDER) ++stats.feeders;
		else ++stats.predators;
		energy_sum += agent.m_energy;
		stats.mean_age += static_cast<double>(agent.m_age);
		const auto state_index = static_cast<std::size_t>(agent.m_state);
		if(state_index < stats.state_counts.size()) ++stats.state_counts[state_index];
		const auto tier_index = static_cast<std::size_t>(agent.brain_tier());
		if(tier_index < stats.brain_tier_counts.size()) ++stats.brain_tier_counts[tier_index];
	}
	for(const Food& food : m_food)
	{
		if(food.m_food_type == FoodType::VEGETABLE) ++stats.vegetables;
		else ++stats.meat;
	}
	if(!m_agents.empty())
	{
		stats.mean_energy = energy_sum / static_cast<double>(m_agents.size());
		stats.mean_age /= static_cast<double>(m_agents.size());
	}
	return stats;
}

void Simulation::write_telemetry()
{
	if(!m_telemetry_output) return;
	const std::size_t interval = std::max<std::size_t>(1, m_config.telemetry_interval_ticks);
	if(m_tick_count % interval != 0) return;
	const EcologyStats stats = ecology_stats();
	*m_telemetry_output << stats.tick << ',' << stats.feeders << ',' << stats.predators << ','
		<< stats.vegetables << ',' << stats.meat << ',' << stats.total_births << ','
		<< stats.total_deaths << ',' << stats.total_vegetables_eaten << ','
		<< stats.mean_energy << ',' << stats.mean_age;
	for(const std::size_t count : stats.state_counts) *m_telemetry_output << ',' << count;
	for(std::size_t tier = static_cast<std::size_t>(BrainTier::SMALL); tier < BRAIN_TIER_COUNT; ++tier)
		*m_telemetry_output << ',' << stats.brain_tier_counts[tier];
	*m_telemetry_output << '\n';
	m_telemetry_output->flush();
	if(!*m_telemetry_output)
	{
		// Telemetry must not invalidate an otherwise healthy simulation tick.
		std::cerr << "Telemetry write failed; disabling telemetry: "
		          << m_config.telemetry_path << '\n';
		m_telemetry_output.reset();
	}
}

void Simulation::set_brain_tier(const BrainTier tier)
{
	if(tier == BrainTier::CUSTOM) return;
	std::lock_guard lock(m_data_mutex);
	m_config.brain_tier = tier;
}

std::size_t Simulation::worker_count() const
{
	return m_thread_pool->size();
}

Simulation::~Simulation() = default;

state_checksum::type Simulation::state_checksum() const
{
	auto hash = ::state_checksum::initial;
	::state_checksum::add(hash, m_agents.size());
	for(const Agent& agent : m_agents) ::state_checksum::add(hash, agent.checksum());
	::state_checksum::add(hash, m_food.size());
	for(const Food& food : m_food)
	{
		::state_checksum::add(hash, food.m_food_type);
		::state_checksum::add(hash, food.position.x);
		::state_checksum::add(hash, food.position.y);
		::state_checksum::add(hash, food.position.z);
	}
	::state_checksum::add(hash, m_food_spawn_timer);
	::state_checksum::add(hash, m_tick_count);
	::state_checksum::add(hash, m_total_births);
	::state_checksum::add(hash, m_total_deaths);
	::state_checksum::add(hash, m_total_vegetables_eaten);
	::state_checksum::add(hash, m_vegetables_eaten_since_balance);
	return hash;
}
