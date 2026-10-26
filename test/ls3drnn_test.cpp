// Unit tests for ls3DRNN: process_step shape/clamping, prediction dim,
// copy independence, training keeps state finite.

#include "ls3DRNN.h"
#include "BrainCatalog.h"
#include "test_framework.h"

TEST_CASE(process_step_returns_finite_transform_with_clamped_translation)
{
	ls3DRNN brain(10, {8, 8});
	const vector<float> command(10, 0.5f);

	const glm::mat4& t = brain.process_step(command, 2.0f);
	for(int i = 0; i < 4; ++i)
		for(int j = 0; j < 4; ++j)
			CHECK(std::isfinite(t[i][j]));

	const glm::vec3 translation(t[3]);
	CHECK(glm::length(translation) <= 2.0f + 1e-5f); // max_speed clamp
}

TEST_CASE(get_prediction_has_seven_params)
{
	ls3DRNN brain(10, {6});
	const vector<float> command(10, -0.25f);
	const vector<float> params = brain.get_prediction(command);
	CHECK(params.size() == 7); // 3 translation + 4 quaternion
	for(float v : params) CHECK(std::isfinite(v));
}

TEST_CASE(copy_constructor_is_deep)
{
	ls3DRNN a(8, {5, 5});
	const vector<float> command(8, 0.1f);
	a.process_step(command, 1.0f);

	ls3DRNN b(a); // deep copy via copy ctor
	const glm::mat4 before = b.get_state();

	// Mutating the original must not touch the copy's state or behavior.
	const vector<float> different(8, -0.9f);
	a.process_step(different, 1.0f);
	CHECK(b.get_state() == before);
}

TEST_CASE(train_and_apply_keeps_prediction_finite)
{
	ls3DRNN brain(10, {8, 8});
	const vector<float> command(10, 0.3f);
	const vector<float> target(7, 0.1f);

	for(int i = 0; i < 5; ++i)
	{
		brain.train_step(command, target, 1.0f);
	}
	brain.apply_and_clear_gradients(0.01f, 5);

	const vector<float> params = brain.get_prediction(command);
	for(float v : params) CHECK(std::isfinite(v));
}

TEST_CASE(batched_training_matches_scalar_reference_exactly)
{
	constexpr std::size_t count = 4;
	std::vector<ls3DRNN> scalar;
	std::vector<ls3DRNN> batched;
	std::vector<ls3DRNN*> brains;
	std::vector<const vector<float>*> commands;
	std::vector<const vector<float>*> targets;
	std::vector<vector<float>> input_values;
	std::vector<vector<float>> target_values;
	scalar.reserve(count);
	batched.reserve(count);
	brains.reserve(count);
	commands.reserve(count);
	targets.reserve(count);
	input_values.reserve(count);
	target_values.reserve(count);
	for(std::size_t i = 0; i < count; ++i)
	{
		scalar.emplace_back(5, movement_hidden_dims(BrainTier::SMALL), static_cast<std::uint32_t>(100 + i));
		batched.emplace_back(5, movement_hidden_dims(BrainTier::SMALL), static_cast<std::uint32_t>(100 + i));
		input_values.emplace_back(5, 0.1f * static_cast<float>(i + 1));
		target_values.emplace_back(7, -0.05f * static_cast<float>(i));
		brains.push_back(&batched.back());
		commands.push_back(&input_values.back());
		targets.push_back(&target_values.back());
		scalar[i].train_step(input_values[i], target_values[i], 0.25f);
	}
	ls3DRNN::train_batch(brains, commands, targets, 0.25f);
	for(std::size_t i = 0; i < count; ++i)
	{
		CHECK(scalar[i].checksum() == batched[i].checksum());
		const vector<float> expected = scalar[i].get_prediction(input_values[i]);
		const vector<float> actual = batched[i].get_prediction(input_values[i]);
		for(std::size_t j = 0; j < expected.size(); ++j) CHECK_NEAR(actual[j], expected[j], 0.0f);
		scalar[i].apply_and_clear_gradients(0.01f, 1);
		batched[i].apply_and_clear_gradients(0.01f, 1);
		scalar[i].process_step(input_values[i], 1.0f);
		batched[i].process_step(input_values[i], 1.0f);
		CHECK(scalar[i].checksum() == batched[i].checksum());
	}
}

TEST_MAIN
