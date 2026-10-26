#ifndef LSRNN_H
#define LSRNN_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <random>
#include <limits>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <queue>
#include <functional>
#include <cstdint>

#include "StateChecksum.h"

using namespace std;

// Enum to define the task for the RNN (Classification or Regression)
enum class Task {
	CLASSIFICATION,
	REGRESSION
};


class lsRNN
{
public:
	// All math is done in single precision: the simulation tolerates it,
	// it halves memory traffic, and it vectorizes better.
	using value_type = float;

	explicit lsRNN(size_t input_dim, size_t hidden_dim, size_t output_dim, Task task, std::uint32_t random_seed = std::random_device{}());

	// Forward pass. Returns a reference to an internal output buffer that is
	// overwritten by the next forward_pass call - copy it if you need to keep it.
	const vector<float>& forward_pass(const vector<float>& input_vector);
	void backpropagate(
		const vector<vector<float>>& inputs,
		const vector<vector<float>>& outputs,
		const vector<vector<float>>& expected_outputs
	);

	// Single-step training convenience: reset_state + forward + backpropagate
	// for a length-1 sequence without any per-call vector allocations.
	void train_step(const vector<float>& input, const vector<float>& expected_output);

	void apply_and_clear_gradients(float learning_rate, size_t batch_size);

	// Loss functions are now part of the class
	float get_loss_cross_entropy(const vector<float>& predicted, const vector<float>& expected);
	float get_loss_mse(const vector<float>& predicted, const vector<float>& expected);

	void reset_state();
	state_checksum::type checksum() const;

	// Utils.
	pair<float, float> find_min_max(const vector<float>& data);
	vector<float> normalize_data(const vector<float>& data, float min_val, float max_val);
	float denormalize_value(float normalized_val, float min_val, float max_val);

	// debug
	void print_weights();

private:
	// Utility functions

	// Activations
	void softmax(vector<float>& vec);
	void normalize(vector<float>& vec);

	// Dimensions
	size_t input_dim;
	size_t hidden_dim;
	size_t output_dim;
	Task m_task; // Private member to store the task type

	// Weights and biases, stored as flat row-major arrays.
	// Element (i, j) of W_xh lives at W_xh[i * input_dim + j], etc.
	// One allocation per matrix instead of a vector of vectors.
	vector<float> W_xh; // hidden_dim * input_dim
	vector<float> W_hh; // hidden_dim * hidden_dim
	vector<float> W_ho; // output_dim * hidden_dim

	vector<float> b_h; // hidden_dim
	vector<float> b_o; // output_dim

	// Reused scratch buffers: forward_pass/backpropagate write into these
	// instead of allocating fresh vectors every call.
	mutable vector<float> m_h_new;   // hidden_dim
	mutable vector<float> m_output;  // output_dim
	mutable vector<float> m_d_output;    // output_dim
	mutable vector<float> m_d_h;         // hidden_dim
	mutable vector<float> m_d_h_next;    // hidden_dim
	mutable vector<float> m_d_h_raw;     // hidden_dim
	mutable vector<vector<float>> m_dW_xh; // hidden_dim x input_dim
	mutable vector<vector<float>> m_dW_hh; // hidden_dim x hidden_dim
	mutable vector<vector<float>> m_dW_ho; // output_dim x hidden_dim
	mutable vector<float> m_db_h;        // hidden_dim
	mutable vector<float> m_db_o;        // output_dim

	// Reused length-1 sequence wrappers for train_step().
	mutable vector<vector<float>> m_seq_input{ vector<float>{} };
	mutable vector<vector<float>> m_seq_output{ vector<float>{} };
	mutable vector<vector<float>> m_seq_expected{ vector<float>{} };

	// State and history. The history is bounded: without a cap, forward_pass
	// appends forever and long-lived agents leak memory. backpropagate()
	// always reads the LAST inputs.size() steps of history.
	vector<float> h;
	// Physical storage is retained across reset_state() calls. The logical
	// ring below avoids both per-tick allocations and large erase/memmove work.
	vector<vector<float>> m_h_history;
	size_t m_h_history_size = 0;
	size_t m_h_history_head = 0; // next physical slot to write when wrapping
	static constexpr size_t MAX_HISTORY = 1024; // steps; older entries are dropped

	// Members for gradient accumulation (same flat layouts as the weights)
	vector<float> m_dW_xh_acc;
	vector<float> m_dW_hh_acc;
	vector<float> m_dW_ho_acc;
	vector<float> m_db_h_acc;
	vector<float> m_db_o_acc;
};

#endif // LSRNN_H
