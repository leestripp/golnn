// Unit tests for lsRNN: normalization round-trips, forward shape,
// single-step training reduces classification loss, finite state.

#include "lsRNN.h"
#include "test_framework.h"

TEST_CASE(find_min_max_handles_positive_and_negative)
{
	lsRNN rnn(3, 4, 2, Task::CLASSIFICATION);
	const auto [mn, mx] = rnn.find_min_max({5.0f, -3.0f, 9.0f, 0.0f});
	CHECK_NEAR(mn, -3.0f, 1e-6f);
	CHECK_NEAR(mx, 9.0f, 1e-6f);
}

TEST_CASE(normalize_denormalize_roundtrip)
{
	lsRNN rnn(3, 4, 2, Task::CLASSIFICATION);
	const vector<float> data = {1.0f, 2.5f, -4.0f, 10.0f};
	const auto [mn, mx] = rnn.find_min_max(data);
	const vector<float> norm = rnn.normalize_data(data, mn, mx);

	CHECK_NEAR(norm[2], 0.0f, 1e-6f);          // data[2] = -4 is the min -> 0
	CHECK_NEAR(norm[3], 1.0f, 1e-6f);          // data[3] = 10 is the max -> 1

	for(size_t i = 0; i < data.size(); ++i)
	{
		CHECK_NEAR(rnn.denormalize_value(norm[i], mn, mx), data[i], 1e-4f);
	}
}

TEST_CASE(forward_pass_output_shape_and_finiteness)
{
	lsRNN rnn(6, 8, 4, Task::CLASSIFICATION);
	const vector<float> input(6, 0.5f);
	const vector<float>& out = rnn.forward_pass(input);

	CHECK(out.size() == 4);
	float sum = 0.0f;
	for(float v : out)
	{
		CHECK(std::isfinite(v));
		CHECK(v >= 0.0f && v <= 1.0f); // softmax probabilities
		sum += v;
	}
	CHECK_NEAR(sum, 1.0f, 1e-4f);
}

TEST_CASE(train_step_reduces_classification_loss)
{
	lsRNN rnn(4, 12, 3, Task::CLASSIFICATION);
	const vector<float> input = {1.0f, 0.0f, 0.5f, -0.5f};
	const vector<float> target = {0.0f, 1.0f, 0.0f}; // class 1

	auto loss_now = [&]() {
		const vector<float>& pred = rnn.forward_pass(input);
		return rnn.get_loss_cross_entropy(pred, target);
	};

	const float initial = loss_now();
	for(int i = 0; i < 60; ++i)
	{
		rnn.train_step(input, target);
		rnn.apply_and_clear_gradients(0.05f, 1);
	}
	const float after = loss_now();

	CHECK(std::isfinite(after));
	CHECK(after < initial);
}

TEST_CASE(regression_train_step_reduces_mse)
{
	lsRNN rnn(3, 10, 2, Task::REGRESSION);
	const vector<float> input = {0.25f, -0.5f, 1.0f};
	const vector<float> target = {0.8f, -0.3f};

	auto mse_now = [&]() {
		const vector<float>& pred = rnn.forward_pass(input);
		return rnn.get_loss_mse(pred, target);
	};

	const float initial = mse_now();
	for(int i = 0; i < 80; ++i)
	{
		rnn.train_step(input, target);
		rnn.apply_and_clear_gradients(0.02f, 1);
	}
	const float after = mse_now();

	CHECK(std::isfinite(after));
	CHECK(after < initial);
}

TEST_CASE(history_ring_supports_latest_step_after_wraparound)
{
	lsRNN rnn(3, 5, 2, Task::REGRESSION, 123);
	const vector<float> input = {0.25f, -0.5f, 1.0f};
	const vector<float> target = {0.8f, -0.3f};
	for(int i = 0; i < 1100; ++i) rnn.forward_pass(input);

	const vector<float> output = rnn.forward_pass(input);
	// backpropagate reads only the requested trailing sequence, even though
	// older states have been overwritten by the bounded ring.
	rnn.backpropagate({input}, {output}, {target});
	rnn.apply_and_clear_gradients(0.001f, 1);
	for(const float value : rnn.forward_pass(input)) CHECK(std::isfinite(value));
}

TEST_MAIN
