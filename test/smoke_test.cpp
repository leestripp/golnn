// golnn smoke test: construct a world, tick it, sanity-check invariants.
// Deliberately framework-free; wired into CTest as the "smoke" test.

#include "Simulation.h"

#include <iostream>

int main()
{
	Simulation sim(50, 50, 100.0f);
	for(int i = 0; i < 100; ++i)
	{
		sim.tick();
	}

	// Sanity bounds: population must respect the sim caps with headroom,
	// and the world must still contain data structures.
	const size_t agent_count = sim.m_agents.size();
	const size_t food_count = sim.m_food.size();
	bool ok = true;

	if(agent_count == 0 || agent_count > 2000)
	{
		std::cerr << "smoke: agent count out of range: " << agent_count << "\n";
		ok = false;
	}
	if(food_count > 2000)
	{
		std::cerr << "smoke: food count out of range: " << food_count << "\n";
		ok = false;
	}

	for(const auto& agent : sim.m_agents)
	{
		if(!std::isfinite(agent.m_energy) || !std::isfinite(agent.m_transform[3][0]))
		{
			std::cerr << "smoke: non-finite agent state (energy or transform)\n";
			ok = false;
			break;
		}
	}

	if(ok)
	{
		std::cout << "smoke: OK after 100 ticks, agents=" << agent_count
		          << " food=" << food_count << "\n";
	}
	return ok ? 0 : 1;
}
