#include "Simulation.h"
#include "test_framework.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

TEST_CASE(adaptive_regrowth_provides_bounded_emergency_vegetation)
{
	SimulationConfig config;
	config.seed = 81;
	config.worker_count = 1;
	config.predator_spawn_probability = 0.0f;
	config.max_food = 20;
	config.emergency_food_spawn = 3;
	Simulation sim(12, 0, 100.0f, config);
	CHECK(sim.m_food.empty());
	sim.tick();
	const EcologyStats stats = sim.ecology_stats();
	CHECK(stats.feeders == 12);
	CHECK(stats.predators == 0);
	CHECK(stats.vegetables > 0);
	CHECK(stats.vegetables <= 20);
}

TEST_CASE(fixed_regrowth_mode_remains_available)
{
	SimulationConfig config;
	config.seed = 81;
	config.worker_count = 1;
	config.predator_spawn_probability = 0.0f;
	config.food_regrowth_mode = FoodRegrowthMode::FIXED;
	config.food_spawn_interval_ticks = 300;
	config.food_spawn_amount = 0;
	Simulation sim(12, 0, 100.0f, config);
	for(int i = 0; i < 10; ++i) sim.tick();
	CHECK(sim.m_food.empty());
}

TEST_CASE(csv_telemetry_contains_ecology_rows)
{
	const std::filesystem::path path = std::filesystem::temp_directory_path() / "golnn_ecology_test.csv";
	{
		SimulationConfig config;
		config.seed = 19;
		config.worker_count = 1;
		config.telemetry_path = path.string();
		config.telemetry_interval_ticks = 1;
		Simulation sim(10, 10, 50.0f, config);
		sim.tick();
		sim.tick();
	}
	std::ifstream input(path);
	const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
	CHECK(contents.find("tick,feeders,predators,vegetables,meat") == 0);
	CHECK(contents.find("\n1,") != std::string::npos);
	CHECK(contents.find("\n2,") != std::string::npos);
	std::filesystem::remove(path);
}

TEST_MAIN
