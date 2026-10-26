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

TEST_CASE(vegetation_spread_forms_local_patches)
{
	SimulationConfig config;
	config.worker_count = 1;
	config.max_food = 100;
	config.vegetation_cluster_probability = 1.0f;
	config.vegetation_cluster_radius = 5.0f;
	Simulation sim(0, 0, 100.0f, config);
	sim.m_food.push_back({FoodType::VEGETABLE, glm::vec3(10.0f, -5.0f, 2.0f)});
	sim.spawn_new_food(40, 0.8f);
	CHECK(sim.m_food.size() == 41);
	bool found_spread = false;
	for(std::size_t i = 1; i < sim.m_food.size(); ++i)
	{
		bool has_local_parent = false;
		for(std::size_t j = 0; j < i; ++j)
		{
			const float distance = glm::distance(sim.m_food[i].position, sim.m_food[j].position);
			has_local_parent = has_local_parent || distance <= 5.0f + 1e-4f;
		}
		CHECK(has_local_parent);
		found_spread = found_spread
			|| glm::distance(sim.m_food[i].position, glm::vec3(10.0f, -5.0f, 2.0f)) > 0.1f;
	}
	CHECK(found_spread);
}

TEST_MAIN
