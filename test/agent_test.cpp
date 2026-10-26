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

TEST_CASE(catalog_tiers_are_used_and_inherited)
{
	SimulationConfig config;
	config.seed = 44;
	config.worker_count = 1;
	config.brain_tier = BrainTier::MEDIUM;
	Simulation sim(1, 0, 100.0f, config);
	Agent& parent = sim.m_agents.front();
	CHECK(parent.brain_tier() == BrainTier::MEDIUM);
	CHECK(parent.m_brain.get_hidden_layer_dims() == movement_hidden_dims(BrainTier::MEDIUM));
	CHECK(parent.get_state_brain_hidden_dim() == state_hidden_dim(BrainTier::MEDIUM));
	parent.m_energy = 200.0f;
	Agent child = parent.reproduce();
	CHECK(child.brain_tier() == BrainTier::MEDIUM);
	CHECK(child.m_brain.get_hidden_layer_dims() == movement_hidden_dims(BrainTier::MEDIUM));
	CHECK(child.get_state_brain_hidden_dim() == state_hidden_dim(BrainTier::MEDIUM));
}

TEST_CASE(runtime_tier_selection_creates_mixed_population)
{
	SimulationConfig config;
	config.seed = 45;
	config.worker_count = 1;
	config.brain_tier = BrainTier::SMALL;
	config.predator_spawn_probability = 0.0f;
	Simulation sim(1, 0, 100.0f, config);
	sim.set_brain_tier(BrainTier::MEDIUM);
	sim.spawn_new_agents(1);
	sim.set_brain_tier(BrainTier::LARGE);
	sim.spawn_new_agents(1);
	CHECK(sim.m_agents.size() == 3);
	CHECK(sim.m_agents[0].brain_tier() == BrainTier::SMALL);
	CHECK(sim.m_agents[1].brain_tier() == BrainTier::MEDIUM);
	CHECK(sim.m_agents[2].brain_tier() == BrainTier::LARGE);
	sim.tick();
	const EcologyStats stats = sim.ecology_stats();
	CHECK(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::SMALL)] == 1);
	CHECK(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::MEDIUM)] == 1);
	CHECK(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::LARGE)] == 1);
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
	CHECK(child.m_brain.get_hidden_layer_dims() == parent_dims);
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

TEST_CASE(perception_v2_tracks_goal_motion_and_compact_state_inputs)
{
	CHECK(perception::VERSION == 3);
	CHECK(Agent::PERCEPTION_VECTOR_SIZE == 50);
	CHECK(Agent::STATE_BRAIN_INPUT_DIM == 18);
	Simulation sim(0, 0, 100.0f);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 55.0f));
	sim.m_food.push_back({FoodType::VEGETABLE, glm::vec3(0.0f, 0.0f, 5.0f)});
	sim.rebuild_spatial_grids();
	Agent& feeder = sim.m_agents[0];
	feeder.m_prev_transform = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
	feeder.update(sim);

	CHECK_NEAR(feeder.m_perception[perception::VELOCITY], 1.0f, 1e-5f);
	CHECK_NEAR(feeder.m_perception[perception::HEADING + 2], 1.0f, 1e-5f);
	CHECK_NEAR(feeder.m_perception[perception::GOAL_DIRECTION + 2], 1.0f, 1e-4f);
	CHECK_NEAR(feeder.m_perception[perception::GOAL_ALIGNMENT], 1.0f, 1e-4f);
	CHECK_NEAR(feeder.m_perception[perception::FOOD_DENSITY], 0.1f, 1e-5f);
	CHECK(feeder.m_perception[perception::ENERGY_TREND] < 0.0f);
	CHECK(feeder.m_state_input_scratch[perception::STATE_GOAL_ALIGNMENT]
		== feeder.m_perception[perception::GOAL_ALIGNMENT]);
	CHECK(feeder.m_state_input_scratch[perception::STATE_FOOD_DENSITY]
		== feeder.m_perception[perception::FOOD_DENSITY]);
}

TEST_CASE(vision_has_a_rear_blind_spot_but_scent_is_omnidirectional)
{
	Simulation sim(0, 0, 100.0f);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 50.0f));
	// Identity faces +z; this food is directly behind and outside close range.
	sim.m_food.push_back({FoodType::VEGETABLE, glm::vec3(0.0f, 0.0f, -5.0f)});
	sim.rebuild_spatial_grids();
	Agent& feeder = sim.m_agents[0];
	feeder.update(sim);
	CHECK(feeder.m_perception[perception::FOOD_DISTANCE] == 0.0f);
	CHECK_NEAR(feeder.m_perception[perception::SCENT_DIRECTION + 2], -1.0f, 1e-4f);
	CHECK(feeder.m_perception[perception::SCENT_INTENSITY] > 0.0f);
	CHECK(feeder.m_state == Agent::State::ROAM);
}

TEST_CASE(boundary_proximity_is_continuous_and_normalized)
{
	Simulation sim(0, 0, 100.0f);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(49.5f, 0.0f, 0.0f), 100.0f));
	sim.rebuild_spatial_grids();
	Agent& feeder = sim.m_agents[0];
	feeder.update(sim);
	CHECK_NEAR(feeder.m_perception[perception::BOUNDARY_PROXIMITY], 0.95f, 1e-4f);
	CHECK(feeder.m_perception[perception::BOUNDARY_DIRECTION] < 0.0f);
	CHECK(feeder.m_state_input_scratch[perception::STATE_BOUNDARY_PROXIMITY]
		== feeder.m_perception[perception::BOUNDARY_PROXIMITY]);
}

TEST_CASE(interaction_sounds_are_audible_for_exactly_the_following_tick)
{
	Simulation sim(0, 0, 100.0f);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 100.0f));
	sim.m_sound_events.push_back({glm::vec3(0.0f, 0.0f, 5.0f), 1.0f});
	sim.tick();
	Agent& feeder = sim.m_agents[0];
	CHECK_NEAR(feeder.m_perception[perception::SOUND_DIRECTION + 2], 1.0f, 1e-4f);
	CHECK(feeder.m_perception[perception::SOUND_INTENSITY] > 0.0f);
	sim.tick();
	CHECK(feeder.m_perception[perception::SOUND_INTENSITY] == 0.0f);
}

TEST_CASE(autonomy_ramp_waits_for_the_teacher_phase)
{
	SimulationConfig config;
	config.state_teacher_ticks = 2;
	config.state_autonomy_ramp_ticks = 10;
	config.worker_count = 1;
	Simulation sim(0, 0, 100.0f, config);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 100.0f));
	sim.rebuild_spatial_grids();
	Agent& agent = sim.m_agents[0];
	agent.update(sim);
	agent.update(sim);
	CHECK(agent.m_autonomy == 0.0f);
	agent.update(sim);
	CHECK_NEAR(agent.m_autonomy, 0.1f, 1e-4f);
	CHECK(agent.m_perception[perception::AUTONOMY] == agent.m_autonomy);
	CHECK(agent.m_state_input_scratch[perception::STATE_AUTONOMY] == agent.m_autonomy);
}

TEST_CASE(network_exploration_can_select_all_new_states)
{
	SimulationConfig config;
	config.state_teacher_ticks = 0;
	config.state_autonomy_ramp_ticks = 1;
	config.state_exploration_rate = 1.0f;
	config.worker_count = 1;
	Simulation sim(0, 0, 100.0f, config);
	sim.m_agents.push_back(make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 100.0f));
	sim.rebuild_spatial_grids();
	Agent& agent = sim.m_agents[0];
	agent.m_autonomy = 1.0f;
	std::array<bool, 4> seen{};
	bool rest_has_zero_translation = false;
	for(int i = 0; i < 64; ++i)
	{
		agent.update(sim);
		const auto state = static_cast<std::size_t>(agent.m_state);
		if(state >= 4 && state < 8) seen[state - 4] = true;
		if(agent.m_state == Agent::State::REST)
		{
			const auto& target = agent.movement_target();
			rest_has_zero_translation = rest_has_zero_translation
				|| (std::fabs(target[0]) < 1e-5f && std::fabs(target[1]) < 1e-5f
					&& std::fabs(target[2]) < 1e-5f);
		}
	}
	for(const bool state_seen : seen) CHECK(state_seen);
	CHECK(rest_has_zero_translation);
}

TEST_CASE(reproduction_preserves_autonomy)
{
	Agent parent = make_agent(Agent::AgentType::FEEDER, glm::vec3(0.0f), 200.0f);
	parent.m_autonomy = 0.7f;
	const Agent child = parent.reproduce();
	CHECK_NEAR(child.m_autonomy, 0.7f, 1e-5f);
}

TEST_MAIN
