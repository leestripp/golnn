// Seeded worlds must replay exactly, including brain weights and gradients.

#include "Simulation.h"
#include "test_framework.h"

TEST_CASE(same_seed_replays_identical_worlds)
{
	SimulationConfig config;
	config.seed = 20261026;
	Simulation first(80, 100, 100.0f, config);
	Simulation second(80, 100, 100.0f, config);
	CHECK(first.state_checksum() == second.state_checksum());

	for(int tick = 0; tick < 120; ++tick)
	{
		first.tick();
		second.tick();
		CHECK(first.state_checksum() == second.state_checksum());
	}
}

TEST_CASE(serial_and_parallel_workers_replay_identically)
{
	SimulationConfig serial_config;
	serial_config.seed = 99173;
	serial_config.worker_count = 1;
	SimulationConfig parallel_config = serial_config;
	parallel_config.worker_count = 4;
	Simulation serial(100, 100, 100.0f, serial_config);
	Simulation parallel(100, 100, 100.0f, parallel_config);
	CHECK(serial.state_checksum() == parallel.state_checksum());
	for(int tick = 0; tick < 120; ++tick)
	{
		serial.tick();
		parallel.tick();
		CHECK(serial.state_checksum() == parallel.state_checksum());
	}
}

TEST_CASE(batched_and_scalar_movement_training_replay_identically)
{
	SimulationConfig scalar_config;
	scalar_config.seed = 7331;
	scalar_config.worker_count = 4;
	scalar_config.use_batched_brains = false;
	SimulationConfig batched_config = scalar_config;
	batched_config.use_batched_brains = true;
	Simulation scalar(100, 100, 100.0f, scalar_config);
	Simulation batched(100, 100, 100.0f, batched_config);
	CHECK(scalar.state_checksum() == batched.state_checksum());
	for(int tick = 0; tick < 120; ++tick)
	{
		scalar.tick();
		batched.tick();
		CHECK(scalar.state_checksum() == batched.state_checksum());
	}
}

TEST_CASE(different_seeds_create_different_worlds)
{
	SimulationConfig first_config;
	first_config.seed = 1;
	SimulationConfig second_config;
	second_config.seed = 2;
	Simulation first(40, 50, 100.0f, first_config);
	Simulation second(40, 50, 100.0f, second_config);
	CHECK(first.state_checksum() != second.state_checksum());
}

TEST_CASE(configured_caps_and_spawn_ratio_are_respected)
{
	SimulationConfig config;
	config.seed = 7;
	config.max_agents = 12;
	config.max_food = 9;
	config.predator_spawn_probability = 1.0f;
	config.food_spawn_interval_ticks = 1;
	config.food_spawn_amount = 20;
	Simulation sim(30, 30, 50.0f, config);
	CHECK(sim.m_agents.size() == 12);
	CHECK(sim.m_food.size() == 9);
	for(const Agent& agent : sim.m_agents)
		CHECK(agent.m_type == Agent::AgentType::PREDATOR);
	sim.spawn_new_agents(10);
	sim.spawn_new_food(10);
	CHECK(sim.m_agents.size() == 12);
	CHECK(sim.m_food.size() == 9);
	sim.tick();
	CHECK(sim.m_agents.size() <= 12);
	CHECK(sim.m_food.size() <= 9);
}

TEST_MAIN
