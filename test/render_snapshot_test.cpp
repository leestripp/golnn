#include "Simulation.h"
#include "test_framework.h"

#include <atomic>
#include <stop_token>
#include <thread>

TEST_CASE(snapshots_are_immutable_and_track_ticks)
{
	SimulationConfig config;
	config.seed = 51;
	config.worker_count = 1;
	Simulation sim(20, 10, 100.0f, config);
	const auto initial = sim.render_snapshot();
	CHECK(initial != nullptr);
	CHECK(initial->tick == 0);
	CHECK(initial->agents.size() == 20);
	CHECK(initial->food.size() == 10);
	CHECK(initial->publication_time_seconds > 0.0);
	const glm::mat4 initial_transform = initial->agents.front().transform;

	for(int i = 0; i < 4; ++i) sim.tick();
	const auto latest = sim.render_snapshot();
	CHECK(latest != initial);
	CHECK(latest->tick == 4);
	CHECK(latest->stats.tick == 4);
	CHECK(initial->tick == 0);
	CHECK(initial->agents.front().transform == initial_transform);
	CHECK(!latest->agents.empty());
	CHECK(initial->agents.size() == 20);
}

TEST_CASE(snapshot_selection_and_concurrent_readers_are_safe)
{
	SimulationConfig config;
	config.seed = 52;
	config.worker_count = 2;
	Simulation sim(30, 10, 100.0f, config);
	sim.set_brain_tier(BrainTier::MEDIUM);
	std::atomic<bool> reader_ok = true;
	std::jthread reader([&](const std::stop_token stop_token) {
		std::size_t previous_tick = 0;
		while(!stop_token.stop_requested())
		{
			const auto snapshot = sim.render_snapshot();
			if(!snapshot || snapshot->tick < previous_tick) reader_ok = false;
			if(snapshot) previous_tick = snapshot->tick;
		}
	});
	for(int i = 0; i < 40; ++i) sim.tick();
	reader.request_stop();
	reader.join();
	CHECK(reader_ok.load());
	const auto snapshot = sim.render_snapshot();
	CHECK(snapshot->tick == 40);
	CHECK(snapshot->selected_brain_tier == BrainTier::MEDIUM);
	CHECK(snapshot->worker_count == 2);
}

TEST_MAIN
