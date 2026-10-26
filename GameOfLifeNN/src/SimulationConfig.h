#ifndef SIMULATION_CONFIG_H
#define SIMULATION_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "BrainCatalog.h"

enum class FoodRegrowthMode
{
	FIXED,
	ADAPTIVE
};

// Plain simulation settings. A default-constructed config preserves the
// original non-repeating world behavior; pass an explicit seed for replay.
struct SimulationConfig
{
	std::uint32_t seed = std::random_device{}();
	std::size_t max_agents = 1000;
	std::size_t max_food = 1000;
	float predator_spawn_probability = 0.1f;
	std::size_t food_spawn_interval_ticks = 300;
	std::size_t food_spawn_amount = 5;
	// 0 selects the machine's hardware concurrency; 1 is the serial fallback.
	std::size_t worker_count = 0;

	BrainTier brain_tier = BrainTier::SMALL;
	bool use_batched_brains = true;
	// GUI builds publish immutable render snapshots; headless users can opt out.
	bool publish_render_snapshots = true;

	FoodRegrowthMode food_regrowth_mode = FoodRegrowthMode::ADAPTIVE;
	float target_vegetation_per_feeder = 0.5f;
	std::size_t food_balance_interval_ticks = 30;
	std::size_t max_food_per_balance = 50;
	std::size_t emergency_food_spawn = 2;

	// Empty path disables telemetry. Otherwise write one CSV row per interval.
	std::string telemetry_path;
	std::size_t telemetry_interval_ticks = 100;
};

#endif // SIMULATION_CONFIG_H
