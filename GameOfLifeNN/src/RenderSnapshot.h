#ifndef RENDER_SNAPSHOT_H
#define RENDER_SNAPSHOT_H

#include <array>
#include <cstddef>
#include <vector>
#include <glm/glm.hpp>

#include "BrainCatalog.h"

struct EcologyStats
{
	std::size_t tick = 0;
	std::size_t feeders = 0;
	std::size_t predators = 0;
	std::size_t vegetables = 0;
	std::size_t meat = 0;
	std::size_t total_births = 0;
	std::size_t total_deaths = 0;
	std::size_t total_vegetables_eaten = 0;
	double mean_energy = 0.0;
	double mean_age = 0.0;
	std::array<std::size_t, 4> state_counts{};
	std::array<std::size_t, BRAIN_TIER_COUNT> brain_tier_counts{};
};

struct AgentRenderInstance
{
	glm::mat4 transform;
	glm::mat4 previous_transform;
	glm::vec4 color;
};

struct FoodRenderInstance
{
	glm::vec3 position;
	glm::vec3 color;
};

// Renderer-facing data only. Once published, a snapshot is never mutated;
// readers hold it with shared_ptr and need no Simulation locks.
struct WorldSnapshot
{
	std::size_t tick = 0;
	BrainTier selected_brain_tier = BrainTier::SMALL;
	std::size_t worker_count = 1;
	double publication_time_seconds = 0.0;
	EcologyStats stats;
	std::vector<AgentRenderInstance> agents;
	std::vector<FoodRenderInstance> food;
};

#endif // RENDER_SNAPSHOT_H
