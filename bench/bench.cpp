// golnn headless benchmark.
// Runs Simulation::tick() without GTK/GL and reports timing.
// Usage: golnn_bench [ticks] [initial_agents] [initial_food] [world_size]

#include "Simulation.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv)
{
	const size_t ticks  = (argc > 1) ? std::stoul(argv[1]) : 1000;
	const size_t agents = (argc > 2) ? std::stoul(argv[2]) : 100;
	const size_t food   = (argc > 3) ? std::stoul(argv[3]) : 50;
	const float  world  = (argc > 4) ? std::stof(argv[4]) : 100.0f;

	std::cout << "bench: ticks=" << ticks
	          << " agents=" << agents
	          << " food=" << food
	          << " world=" << world << "\n";

	Simulation sim(agents, food, world);
	sim.tick(); // warmup + JIT/alloc warm

	const auto t0 = std::chrono::steady_clock::now();
	for(size_t i = 0; i < ticks; ++i)
	{
		sim.tick();
	}
	const auto t1 = std::chrono::steady_clock::now();

	const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	const double ms_per_tick = total_ms / static_cast<double>(ticks);

	std::cout << "result: total_ms=" << total_ms
	          << " ms_per_tick=" << ms_per_tick
	          << " ticks_per_sec=" << (1000.0 / ms_per_tick)
	          << " final_agents=" << sim.m_agents.size()
	          << " final_food=" << sim.m_food.size() << "\n";
	return 0;
}
