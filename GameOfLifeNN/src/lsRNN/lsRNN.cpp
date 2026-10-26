#include "lsRNN.h"

// Utils.

pair<float, float> lsRNN::find_min_max(const vector<float>& data)
{
	// Seed min high and max low so the first real value always wins.
	// (lowest() = most negative, max() = most positive; the original code
	// used min() = smallest positive, which broke all-negative data, and a
	// later "fix" swapped the seeds so nothing ever updated them.)
	float min_val = numeric_limits<float>::max();
	float max_val = numeric_limits<float>::lowest();
	for(float val : data)
	{
		if(val < min_val) min_val = val;
		if(val > max_val) max_val = val;
	}
	return {min_val, max_val};
}

// Normalize a vector using Min-Max scaling
vector<float> lsRNN::normalize_data(const vector<float>& data, float min_val, float max_val)
{
	vector<float> normalized_data(data.size());
	float range = max_val - min_val;

	// Prevent division by zero.
	if(range < 1e-9f)
	{
		return vector<float>(data.size(), 0.0f);
	}

	for(size_t i = 0; i < data.size(); i++)
	{
		normalized_data[i] = (data[i] - min_val) / range;
	}
	return normalized_data;
}

// Denormalize a value back to its original scale
float lsRNN::denormalize_value(float normalized_val, float min_val, float max_val)
{
	float range = max_val - min_val;
	return normalized_val * range + min_val;
}

void lsRNN::softmax(vector<float>& vec)
{
	float max_val = -numeric_limits<float>::infinity();
	for (float val : vec)
	{
		if (val > max_val)
		{
			max_val = val;
		}
	}

	float sum = 0.0f;
	for (size_t i = 0; i < vec.size(); ++i)
	{
		vec[i] = exp(vec[i] - max_val);
		sum += vec[i];
	}

	for (float& val : vec)
	{
		val /= sum;
	}
}

void lsRNN::normalize(vector<float>& vec)
{
	float magnitude = 0.0f;
	for(float val : vec)
	{
		magnitude += val * val;
	}
	magnitude = sqrt(magnitude);

	if(magnitude > 1e-6f)
	{
		for(float& val : vec)
		{
			val /= magnitude;
		}
	}
}


// lsRNN Constructor
lsRNN::lsRNN(size_t in_dim, size_t hid_dim, size_t out_dim, Task task, std::uint32_t random_seed) :
	input_dim(in_dim),
	hidden_dim(hid_dim),
	output_dim(out_dim),
	m_task(task),
	// Flat weight matrices, single allocation each
	W_xh(hid_dim * in_dim),
	W_hh(hid_dim * hid_dim),
	W_ho(out_dim * hid_dim),
	b_h(hid_dim, 0.0f),
	b_o(out_dim, 0.0f),
	// Scratch buffers, allocated once and reused every step
	m_h_new(hid_dim),
	m_output(out_dim),
	m_d_output(out_dim),
	m_d_h(hid_dim),
	m_d_h_next(hid_dim),
	m_d_h_raw(hid_dim),
	m_dW_xh(hid_dim, vector<float>(in_dim, 0.0f)),
	m_dW_hh(hid_dim, vector<float>(hid_dim, 0.0f)),
	m_dW_ho(out_dim, vector<float>(hid_dim, 0.0f)),
	m_db_h(hid_dim, 0.0f),
	m_db_o(out_dim, 0.0f),
	h(hid_dim, 0.0f),
	m_dW_xh_acc(hid_dim * in_dim, 0.0f),
	m_dW_hh_acc(hid_dim * hid_dim, 0.0f),
	m_dW_ho_acc(out_dim * hid_dim, 0.0f),
	m_db_h_acc(hid_dim, 0.0f),
	m_db_o_acc(out_dim, 0.0f)
{
	std::mt19937 rng(random_seed);
	std::uniform_real_distribution<float> weight_dist(-0.5f, 0.5f);
	for (float& val : W_xh) val = weight_dist(rng);
	for (float& val : W_hh) val = weight_dist(rng);
	for (float& val : W_ho) val = weight_dist(rng);

	m_h_history.reserve(64);
	m_h_history.push_back(h);
	m_h_history_size = 1;
	m_h_history_head = 1;
}

// Loss functions are now member functions
void lsRNN::mutate(const float mutation_rate, const float mutation_amount, std::mt19937& rng)
{
	std::uniform_real_distribution<float> chance(0.0f, 1.0f);
	std::uniform_real_distribution<float> delta(-mutation_amount, mutation_amount);
	for(vector<float>* weights : {&W_xh, &W_hh, &W_ho, &b_h, &b_o})
	{
		for(float& weight : *weights)
			if(chance(rng) < mutation_rate) weight += delta(rng);
	}
}

float lsRNN::get_loss_cross_entropy(const vector<float>& predicted, const vector<float>& expected)
{
	float loss = 0.0f;
	for (size_t i = 0; i < predicted.size(); ++i)
	{
		if (expected[i] == 1.0f)
		{
			loss -= log(predicted[i] + 1e-9f); // Add a small value to prevent log(0)
		}
	}
	return loss;
}

float lsRNN::get_loss_mse(const vector<float>& predicted, const vector<float>& expected)
{
	float loss = 0.0f;
	for (size_t i = 0; i < predicted.size(); ++i)
	{
		const float diff = predicted[i] - expected[i];
		loss += diff * diff;
	}
	return loss / static_cast<float>(predicted.size());
}

void lsRNN::reset_state()
{
	fill(h.begin(), h.end(), 0.0f);
	if(m_h_history.empty()) m_h_history.push_back(h);
	else m_h_history[0] = h;
	m_h_history_size = 1;
	m_h_history_head = 1;
}

void lsRNN::print_weights()
{
	cout << "### Hidden Layers:" << endl;
	const size_t rows = hidden_dim;
	for(size_t i = 0; i < rows; ++i) {
		cout << "[";
		const float* row = &W_hh[i * hidden_dim];
		for(size_t j = 0; j < hidden_dim; ++j) {
			cout << fixed << setprecision(4) << row[j] << (j < hidden_dim - 1 ? ", " : "");
		}
		cout << "]" << endl;
	}
}

const vector<float>& lsRNN::forward_pass(const vector<float>& input_vector)
{
	// h_new = tanh(b_h + W_xh.x + W_hh.h), written into the reusable buffer.
	for (size_t i = 0; i < hidden_dim; ++i) {
		float sum = b_h[i];
		const float* wxh_row = &W_xh[i * input_dim];
		for (size_t j = 0; j < input_dim; ++j) {
			sum += wxh_row[j] * input_vector[j];
		}
		const float* whh_row = &W_hh[i * hidden_dim];
		for (size_t j = 0; j < hidden_dim; ++j) {
			sum += whh_row[j] * h[j];
		}
		m_h_new[i] = tanh(sum);
	}

	h = m_h_new;

	// Retain physical slots across calls. Agent::train_step() resets every
	// tick, so reusing slot 1 here is what keeps that path allocation-free.
	if(m_h_history_size == m_h_history.size() && m_h_history.size() < MAX_HISTORY)
	{
		m_h_history.push_back(h);
		++m_h_history_size;
		m_h_history_head = m_h_history_size % MAX_HISTORY;
	}
	else
	{
		m_h_history[m_h_history_head] = h;
		m_h_history_head = (m_h_history_head + 1) % MAX_HISTORY;
		if(m_h_history_size < MAX_HISTORY) ++m_h_history_size;
	}

	// output = b_o + W_ho.h_new
	for (size_t i = 0; i < output_dim; ++i) {
		float sum = b_o[i];
		const float* who_row = &W_ho[i * hidden_dim];
		for (size_t j = 0; j < hidden_dim; ++j) {
			sum += who_row[j] * m_h_new[j];
		}
		m_output[i] = sum;
	}

	// Apply activation function based on the task
	if (m_task == Task::CLASSIFICATION) {
		softmax(m_output);
	}
	// For regression, we use a linear output, so no activation is applied.

	return m_output;
}

void lsRNN::backpropagate(
	const vector<vector<float>>& inputs,
	const vector<vector<float>>& outputs,
	const vector<vector<float>>& expected_outputs
)
{
	const size_t T = inputs.size();
	if (T == 0) return;

	// The history holds h BEFORE each step plus all intermediate states.
	// We only need the last T+1 entries, which lets forward_pass run
	// unbounded batches of steps between backprop calls.
	if (T + 1 > m_h_history_size) {
		throw invalid_argument("lsRNN::backpropagate: input sequence longer than recorded history");
	}
	const size_t base = m_h_history_size - T - 1; // logical index of h BEFORE inputs[0]
	const auto history_at = [&](const size_t logical_index) -> const vector<float>& {
		const size_t physical_index = m_h_history_size == MAX_HISTORY
			? (m_h_history_head + logical_index) % MAX_HISTORY
			: logical_index;
		return m_h_history[physical_index];
	};

	// Zero the reused gradient scratch
	for (auto& row : m_dW_xh) fill(row.begin(), row.end(), 0.0f);
	for (auto& row : m_dW_hh) fill(row.begin(), row.end(), 0.0f);
	for (auto& row : m_dW_ho) fill(row.begin(), row.end(), 0.0f);
	fill(m_db_h.begin(), m_db_h.end(), 0.0f);
	fill(m_db_o.begin(), m_db_o.end(), 0.0f);
	fill(m_d_h_next.begin(), m_d_h_next.end(), 0.0f);

	for(int t = static_cast<int>(T) - 1; t >= 0; --t)
	{
		const vector<float>& h_next = history_at(base + static_cast<size_t>(t) + 1);
		const vector<float>& h_prev = history_at(base + static_cast<size_t>(t));

		// Calculate the output gradient based on the task
		switch (m_task) {
			case Task::CLASSIFICATION:
				// For Softmax + Cross-Entropy Loss, the output gradient is simply predicted - expected
				for(size_t i = 0; i < output_dim; ++i) {
					m_d_output[i] = outputs[t][i] - expected_outputs[t][i];
				}
				break;
			case Task::REGRESSION:
				// For MSE loss, the output gradient is 2 * (predicted - expected)
				for(size_t i = 0; i < output_dim; ++i) {
					m_d_output[i] = 2.0f * (outputs[t][i] - expected_outputs[t][i]) / static_cast<float>(output_dim);
				}
				break;
		}

		for(size_t i = 0; i < output_dim; ++i)
		{
			m_db_o[i] += m_d_output[i];
			for(size_t j = 0; j < hidden_dim; ++j)
			{
				m_dW_ho[i][j] += m_d_output[i] * h_next[j];
			}
		}

		fill(m_d_h.begin(), m_d_h.end(), 0.0f);
		for(size_t i = 0; i < hidden_dim; ++i)
		{
			for(size_t j = 0; j < output_dim; ++j)
			{
				m_d_h[i] += m_d_output[j] * W_ho[j * hidden_dim + i];
			}
			for(size_t j = 0; j < hidden_dim; ++j)
			{
				m_d_h[i] += m_d_h_next[j] * W_hh[j * hidden_dim + i];
			}
		}

		for(size_t i = 0; i < hidden_dim; ++i)
		{
			m_d_h_raw[i] = m_d_h[i] * (1.0f - h_next[i] * h_next[i]);
		}

		for(size_t i = 0; i < hidden_dim; ++i)
		{
			m_db_h[i] += m_d_h_raw[i];
			for(size_t j = 0; j < input_dim; ++j)
			{
				m_dW_xh[i][j] += m_d_h_raw[i] * inputs[t][j];
			}
			for(size_t j = 0; j < hidden_dim; ++j)
			{
				m_dW_hh[i][j] += m_d_h_raw[i] * h_prev[j];
			}
		}

		m_d_h_next = m_d_h_raw;
	}

	constexpr float clip_value = 5.0f;
	auto clip = [clip_value](float val) { return std::clamp(val, -clip_value, clip_value); };
	for (float& val : m_db_o) val = clip(val);
	for (float& val : m_db_h) val = clip(val);
	for (auto& row : m_dW_xh) for(float& val : row) val = clip(val);
	for (auto& row : m_dW_hh) for(float& val : row) val = clip(val);
	for (auto& row : m_dW_ho) for(float& val : row) val = clip(val);

	// Accumulate gradients instead of applying them
	for(size_t i = 0; i < hidden_dim; ++i)
	{
		for(size_t j = 0; j < input_dim; ++j) m_dW_xh_acc[i * input_dim + j] += m_dW_xh[i][j];
		for(size_t j = 0; j < hidden_dim; ++j) m_dW_hh_acc[i * hidden_dim + j] += m_dW_hh[i][j];
		m_db_h_acc[i] += m_db_h[i];
	}

	for(size_t i = 0; i < output_dim; ++i)
	{
		for(size_t j = 0; j < hidden_dim; ++j) m_dW_ho_acc[i * hidden_dim + j] += m_dW_ho[i][j];
		m_db_o_acc[i] += m_db_o[i];
	}
}

void lsRNN::train_step(const vector<float>& input, const vector<float>& expected_output)
{
	reset_state();
	const vector<float>& predicted = forward_pass(input);

	// Delegate to backpropagate() with reused length-1 sequence wrappers so
	// the gradient math (including clipping) is identical to the batched
	// path, without any per-call heap allocations.
	m_seq_input[0] = input;
	m_seq_output[0] = predicted;
	m_seq_expected[0] = expected_output;
	backpropagate(m_seq_input, m_seq_output, m_seq_expected);
}

void lsRNN::apply_and_clear_gradients(float learning_rate, size_t batch_size)
{
	if (batch_size == 0) return;
	const float lr_per_sample = learning_rate / static_cast<float>(batch_size);

	// Apply accumulated gradients to weights
	for(size_t i = 0; i < W_xh.size(); ++i) W_xh[i] -= lr_per_sample * m_dW_xh_acc[i];
	for(size_t i = 0; i < W_hh.size(); ++i) W_hh[i] -= lr_per_sample * m_dW_hh_acc[i];
	for(size_t i = 0; i < W_ho.size(); ++i) W_ho[i] -= lr_per_sample * m_dW_ho_acc[i];
	for(size_t i = 0; i < hidden_dim; ++i)
	{
		b_h[i] -= lr_per_sample * m_db_h_acc[i];
	}
	for(size_t i = 0; i < output_dim; ++i)
	{
		b_o[i] -= lr_per_sample * m_db_o_acc[i];
	}

	// Reset accumulators to zero for the next batch
	fill(m_dW_xh_acc.begin(), m_dW_xh_acc.end(), 0.0f);
	fill(m_dW_hh_acc.begin(), m_dW_hh_acc.end(), 0.0f);
	fill(m_dW_ho_acc.begin(), m_dW_ho_acc.end(), 0.0f);
	fill(m_db_h_acc.begin(), m_db_h_acc.end(), 0.0f);
	fill(m_db_o_acc.begin(), m_db_o_acc.end(), 0.0f);
}

state_checksum::type lsRNN::checksum() const
{
	auto hash = state_checksum::initial;
	state_checksum::add(hash, input_dim);
	state_checksum::add(hash, hidden_dim);
	state_checksum::add(hash, output_dim);
	state_checksum::add(hash, static_cast<std::uint64_t>(m_task));
	for(const auto* values : {&W_xh, &W_hh, &W_ho, &b_h, &b_o, &h,
		&m_dW_xh_acc, &m_dW_hh_acc, &m_dW_ho_acc, &m_db_h_acc, &m_db_o_acc})
	{
		state_checksum::add(hash, std::span<const float>(*values));
	}
	return hash;
}
