// Unit tests for Agent: lifecycle rules (death thresholds, reproduction,
// energy defaults) and the teacher-imposed state machine, driven through
// the public update() API with hand-placed worlds so behavior is
// deterministic despite the random brains.

#include "Agent.h"
#include "Simulation.h"
#include "test_framework.h"

#include <glm/gtx/transform.hpp>

#include <cmath>

namespace
{
	// Small fixed brain; the random weights do not matter for these tests
	// because state is chosen from perception BEFORE the movement step.
	Agent make_agent(Agent::AgentType type, const glm::vec3& pos, float energy)
	{
		ls3DRNN brain(Agent::PERCEPTION_VECTOR_SIZE, {4});
		const glm::mat4 transform = glm::translate(glm::mat4(1.0f), pos);
		return Agent(brain, transform, type, 0.05f, energy);
	}
}

TEST_CASE(isDead_tracks_energy_and_age)
{
	Agent feeder = make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 10.0f);
	CHECK(!feeder.isDead());
	feeder.m_energy = 0.0f;
	CHECK(feeder.isDead());
	feeder.m_energy = -1.0f;
	CHECK(feeder.isDead());

	// MAX_AGE (5000) is a strict limit: exactly 5000 is still alive.
	Agent aged = make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 10.0f);
	aged.m_age = 5000;
	CHECK(!aged.isDead());
	aged.m_age = 5001;
	CHECK(aged.isDead());
}

TEST_CASE(canReproduce_uses_type_specific_thresholds)
{
	Agent feeder = make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 109.99f);
	CHECK(!feeder.canReproduce());
	feeder.m_energy = 110.0f; // FEEDER_ENERGY_TO_REPRODUCE
	CHECK(feeder.canReproduce());

	Agent predator = make_agent(Agent::AgentType::PREDATOR, glm::vec3(0.0f), 129.99f);
	CHECK(!predator.canReproduce());
	predator.m_energy = 130.0f; // PREDATOR_ENERGY_TO_REPRODUCE
	CHECK(predator.canReproduce());
}

TEST_CASE(initial_energy_defaults_by_type)
{
	ls3DRNN brain(Agent::PERCEPTION_VECTOR_SIZE, {4});
	const glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f));
	Agent feeder(brain, t, Agent::AgentType::FEEDER, 0.05f, -1.0f);
	CHECK_NEAR(feeder.m_energy, 100.0f, 1e-4f);
	Agent predator(brain, t, Agent::AgentType::PREDATOR, 0.05f, -1.0f);
	CHECK_NEAR(predator.m_energy, 150.0f, 1e-4f); // 1.5x the feeder baseline
}

TEST_CASE(reproduce_halves_energy_and_preserves_type_and_structure)
{
	Agent parent = make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 200.0f);
	const std::vector<size_t> parent_dims = parent.m_brain.get_hidden_layer_dims();

	Agent child = parent.reproduce();

	CHECK_NEAR(parent.m_energy, 100.0f, 1e-4f);
	CHECK_NEAR(child.m_energy, 100.0f, 1e-4f);
	CHECK(child.m_type == Agent::AgentType::FEEDER);
	CHECK(child.m_age == 0);
	// The child's movement brain is a (mutated) copy of the parent's, so
	// its architecture must match even though the weights differ.
	CHECK(child.get_hidden_layer_dims() == parent_dims);
}

TEST_CASE(reproduce_places_child_near_parent)
{
	Agent parent = make_agent(Agent::AgentType::PREDATOR, glm::vec3(10.0f, -5.0f, 2.0f), 200.0f);
	Agent child = parent.reproduce();

	const glm::vec3 p = glm::vec3(parent.m_transform[3]);
	const glm::vec3 c = glm::vec3(child.m_transform[3]);
	// Child spawns within a 2-unit random offset per axis (see Agent::reproduce).
	CHECK(std::fabs(c.x - p.x) <= 2.0f + 1e-4f);
	CHECK(std::fabs(c.y - p.y) <= 2.0f + 1e-4f);
	CHECK(std::fabs(c.z - p.z) <= 2.0f + 1e-4f);
}

TEST_CASE(feeder_flees_predator_over_all_other_priorities)
{
	Simulation sim(0, 0, 100.0f);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 150.0f));
	sim.m_agents.push_back(make_agent(Agent::AgentType::PREDATOR, glm::vec3(5.0f, 0.0f, 0.0f), 100.0f));
	// Food is in range too, so HUNT would win if FLEE did not have priority.
	sim.m_food.push_back({FoodType::VEGETABLE, glm::vec3(3.0f, 0.0f, 0.0f)});
	sim.rebuild_spatial_grids();

	sim.m_agents[0].update(sim);
	CHECK(sim.m_agents[0].m_state == Agent::State::FLEE);
}

TEST_CASE(feeder_hunts_food_when_hungry_and_safe)
{
	Simulation sim(0, 0, 100.0f);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 50.0f));
	sim.m_food.push_back({FoodType::VEGETABLE, glm::vec3(5.0f, 0.0f, 0.0f)});
	sim.rebuild_spatial_grids();

	sim.m_agents[0].update(sim);
	CHECK(sim.m_agents[0].m_state == Agent::State::HUNT);
}

TEST_CASE(feeder_roams_when_safe_and_satiated)
{
	Simulation sim(0, 0, 100.0f);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 150.0f));
	sim.rebuild_spatial_grids();

	sim.m_agents[0].update(sim);
	CHECK(sim.m_agents[0].m_state == Agent::State::ROAM);
}

TEST_CASE(predator_hunts_feeder_and_meat_but_roams_otherwise)
{
	// Case A: feeder prey in visual range -> HUNT (highest priority).
	Simulation simA(0, 0, 100.0f);
	simA.m_agents.push_back(make_agent(Agent::AgentType::PREDATOR, glm::vec3(0.0f), 100.0f));
	simA.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(5.0f, 0.0f, 0.0f), 80.0f));
	simA.rebuild_spatial_grids();
	simA.m_agents[0].update(simA);
	CHECK(simA.m_agents[0].m_state == Agent::State::HUNT);

	// Case B: only meat in range and hungry -> HUNT.
	Simulation simB(0, 0, 100.0f);
	simB.m_agents.push_back(make_agent(Agent::AgentType::PREDATOR, glm::vec3(0.0f), 50.0f));
	simB.m_food.push_back({FoodType::MEAT, glm::vec3(5.0f, 0.0f, 0.0f)});
	simB.rebuild_spatial_grids();
	simB.m_agents[0].update(simB);
	CHECK(simB.m_agents[0].m_state == Agent::State::HUNT);

	// Case C: satiated and nothing in range -> ROAM.
	Simulation simC(0, 0, 100.0f);
	simC.m_agents.push_back(make_agent(Agent::AgentType::PREDATOR, glm::vec3(0.0f), 200.0f));
	simC.rebuild_spatial_grids();
	simC.m_agents[0].update(simC);
	CHECK(simC.m_agents[0].m_state == Agent::State::ROAM);
}

TEST_CASE(perception_encodes_food_distance_energy_and_state)
{
	Simulation sim(0, 0, 100.0f);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 55.0f));
	sim.m_food.push_back({FoodType::VEGETABLE, glm::vec3(0.0f, 5.0f, 0.0f)});
	sim.rebuild_spatial_grids();

	Agent& feeder = sim.m_agents[0];
	feeder.update(sim);

	CHECK(feeder.m_perception.size() == Agent::PERCEPTION_VECTOR_SIZE);
	// [0-3] closest food: direction +y, distance 5 (captured before movement).
	CHECK_NEAR(feeder.m_perception[1], 1.0f, 1e-3f);
	CHECK_NEAR(feeder.m_perception[3], 5.0f, 1e-3f);
	// [25] energy ratio: 55 / FEEDER_ENERGY_TO_REPRODUCE (110).
	CHECK_NEAR(feeder.m_perception[25], 55.0f / 110.0f, 1e-3f);
	// Hungry feeder with food available -> HUNT one-hot at index 29.
	CHECK_NEAR(feeder.m_perception[29], 1.0f, 1e-6f);
	CHECK_NEAR(feeder.m_perception[27], 0.0f, 1e-6f);
	CHECK_NEAR(feeder.m_perception[28], 0.0f, 1e-6f);
	CHECK_NEAR(feeder.m_perception[30], 0.0f, 1e-6f);
}

TEST_MAIN
