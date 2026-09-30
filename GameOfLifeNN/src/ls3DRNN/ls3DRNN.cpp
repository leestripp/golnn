#include "ls3DRNN.h"

#include <stdexcept>
#include <numeric>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

// Internal MLP Cell.
// Weights are stored as one flat row-major array per layer:
//   layer i has dims m_layer_dims[i] -> m_layer_dims[i+1], and the weight
//   connecting input k to output j lives at m_weights[i][j * in_dim + k].
// This replaces the old vector<vector<vector<float>>>: one allocation per
// layer, contiguous rows, and a layout the auto-vectorizer likes.
class ls3DRNN_MLP_Private
{
public:
	ls3DRNN_MLP_Private(size_t input_dim, size_t output_dim, const vector<size_t>& hidden_dims)
	{
		m_layer_dims.push_back(input_dim);
		m_layer_dims.insert(m_layer_dims.end(), hidden_dims.begin(), hidden_dims.end());
		m_layer_dims.push_back(output_dim);

		const size_t layer_count = m_layer_dims.size() - 1;
		m_weights.resize(layer_count);
		m_biases.resize(layer_count);
		m_accumulated_weight_gradients.resize(layer_count);
		m_accumulated_bias_gradients.resize(layer_count);
		m_layer_inputs.resize(layer_count + 1);  // per-layer activation cache

		srand(static_cast<unsigned int>(time(0)));
		for(size_t i = 0; i < layer_count; i++)
		{
			const size_t in_features = m_layer_dims[i];
			const size_t out_features = m_layer_dims[i + 1];

			vector<float>& weights = m_weights[i];
			weights.resize(out_features * in_features);
			for(float& val : weights)
			{
				val = static_cast<float>(rand()) / RAND_MAX * 0.2f - 0.1f;
			}
			m_biases[i].assign(out_features, 0.0f);
			m_accumulated_weight_gradients[i].assign(out_features * in_features, 0.0f);
			m_accumulated_bias_gradients[i].assign(out_features, 0.0f);
			m_layer_inputs[i].resize(in_features);
		}
		m_layer_inputs[layer_count].resize(output_dim);
	}

	void clear_gradients()
	{
		for (auto& layer_w_grad : m_accumulated_weight_gradients) {
			fill(layer_w_grad.begin(), layer_w_grad.end(), 0.0f);
		}
		for (auto& layer_b_grad : m_accumulated_bias_gradients) {
			fill(layer_b_grad.begin(), layer_b_grad.end(), 0.0f);
		}
	}

	vector<size_t> m_layer_dims;
	vector<vector<float>> m_weights;                        // flat, row-major per layer
	vector<vector<float>> m_biases;
	mutable vector<vector<float>> m_layer_inputs;           // activations per layer (incl. input)
	mutable vector<vector<float>> m_accumulated_weight_gradients;  // flat, row-major per layer
	mutable vector<vector<float>> m_accumulated_bias_gradients;

	// Forward memoization: Agent calls train_step() then process_step() with
	// the SAME input every tick. The second forward is pure waste, so we
	// remember the last input and skip the recompute when it repeats.
	// Invalidated whenever weights change (optimizer_step/mutate).
	mutable vector<float> m_last_input;
	mutable bool m_forward_cache_valid = false;

	vector<float> forward(const vector<float>& input) const
	{
		if (m_forward_cache_valid && m_last_input == input)
		{
			return m_layer_inputs.back();
		}
		m_last_input = input;

		// Cache activations for the backward pass. Buffers are reused; resize
		// is a no-op after the first call with a given geometry.
		m_layer_inputs[0] = input;

		const size_t layer_count = m_weights.size();
		for(size_t i = 0; i < layer_count; i++)
		{
			const size_t in_dim = m_layer_dims[i];
			const size_t out_dim = m_layer_dims[i + 1];
			const vector<float>& prev = m_layer_inputs[i];
			vector<float>& current = m_layer_inputs[i + 1];

			const vector<float>& weights = m_weights[i];
			const vector<float>& biases = m_biases[i];
			for(size_t j = 0; j < out_dim; j++)
			{
				const float* row = &weights[j * in_dim];
				float z = biases[j];
				for(size_t k = 0; k < in_dim; k++)
				{
					z += row[k] * prev[k];
				}
				// tanh on hidden layers only; output layer stays linear
				current[j] = (i < layer_count - 1) ? tanh(z) : z;
			}
		}
		m_forward_cache_valid = true;
		return m_layer_inputs[layer_count];
	}

	void backward(const vector<float>& target_output, float rotation_weight) const
	{
		// The target_output vector is size 7: 3 translation + 4 quaternion.
		const size_t out_dim = target_output.size();
		vector<float> delta_a(out_dim);
		for(size_t i = 0; i < 3; ++i) { delta_a[i] = m_layer_inputs.back()[i] - target_output[i]; }
		for(size_t i = 3; i < 7; ++i) { delta_a[i] = rotation_weight * (m_layer_inputs.back()[i] - target_output[i]); }

		const int layer_count = static_cast<int>(m_weights.size());
		for(int i = layer_count - 1; i >= 0; i--)
		{
			const size_t in_dim = m_layer_dims[i];
			const size_t out_dim_i = m_layer_dims[i + 1];
			const vector<float>& current_activations = m_layer_inputs[i + 1];

			vector<float> delta_z = delta_a;
			// Apply tanh derivative only for hidden layers
			if(i < layer_count - 1)
			{
				for(size_t j = 0; j < out_dim_i; j++)
				{
					const float tanh_derivative = 1.0f - current_activations[j] * current_activations[j];
					delta_z[j] *= tanh_derivative;
				}
			}

			const vector<float>& prev_activations = m_layer_inputs[i];

			vector<float>& w_grads = m_accumulated_weight_gradients[i];
			vector<float>& b_grads = m_accumulated_bias_gradients[i];
			for(size_t j = 0; j < out_dim_i; ++j)
			{
				b_grads[j] += delta_z[j];
				float* grad_row = &w_grads[j * in_dim];
				const float dz = delta_z[j];
				for(size_t k = 0; k < in_dim; ++k)
				{
					grad_row[k] += dz * prev_activations[k];
				}
			}

			if(i > 0)
			{
				vector<float> next_delta_a(in_dim, 0.0f);
				const vector<float>& weights = m_weights[i];
				for(size_t j = 0; j < out_dim_i; ++j)
				{
					const float dz = delta_z[j];
					const float* row = &weights[j * in_dim];
					for(size_t k = 0; k < in_dim; ++k)
					{
						next_delta_a[k] += row[k] * dz;
					}
				}
				delta_a = next_delta_a;
			}
		}
	}

	void optimizer_step(float learning_rate, size_t batch_size)
	{
		if (batch_size == 0) return;
		m_forward_cache_valid = false; // weights are about to change
		const float lr_per_sample = learning_rate / static_cast<float>(batch_size);

		const size_t layer_count = m_weights.size();
		for(size_t i = 0; i < layer_count; ++i)
		{
			vector<float>& weights = m_weights[i];
			vector<float>& biases = m_biases[i];
			const vector<float>& w_grads = m_accumulated_weight_gradients[i];
			const vector<float>& b_grads = m_accumulated_bias_gradients[i];
			for(size_t n = 0; n < weights.size(); ++n)
			{
				weights[n] -= lr_per_sample * w_grads[n];
			}
			for(size_t n = 0; n < biases.size(); ++n)
			{
				biases[n] -= lr_per_sample * b_grads[n];
			}
		}
		clear_gradients();
	}

	// Adjusts weights and biases by a random amount.
	void mutate(float mutation_rate, float mutation_amount)
	{
		m_forward_cache_valid = false; // weights are about to change
		// Helper lambda to generate a random value in [-amount, +amount]
		auto random_mutation = [=]() {
			return (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * mutation_amount;
		};

		const size_t layer_count = m_weights.size();
		for (size_t i = 0; i < layer_count; ++i) {
			for (float& weight : m_weights[i]) {
				if (static_cast<float>(rand()) / RAND_MAX < mutation_rate) {
					weight += random_mutation();
				}
			}
			for (float& bias : m_biases[i]) {
				if (static_cast<float>(rand()) / RAND_MAX < mutation_rate) {
					bias += random_mutation();
				}
			}
		}
	}
};

// ls3DRNN Method Implementations

ls3DRNN::ls3DRNN(size_t command_dim, const vector<size_t>& hidden_dims)
{
	size_t total_input_dim = command_dim + STATE_DIM;
	m_cell = make_unique<ls3DRNN_MLP_Private>(total_input_dim, MLP_OUTPUT_DIM, hidden_dims);
	reset();
}

ls3DRNN::~ls3DRNN() = default;
ls3DRNN::ls3DRNN(ls3DRNN&& other) noexcept = default;
ls3DRNN& ls3DRNN::operator=(ls3DRNN&& other) noexcept = default;

ls3DRNN::ls3DRNN(const ls3DRNN& other) : m_hidden_state(other.m_hidden_state)
{
	m_cell = make_unique<ls3DRNN_MLP_Private>(*other.m_cell);
}

ls3DRNN& ls3DRNN::operator=(const ls3DRNN& other)
{
	if (this != &other)
	{
		m_hidden_state = other.m_hidden_state;
		m_cell = make_unique<ls3DRNN_MLP_Private>(*other.m_cell);
	}
	return *this;
}

// Assemble [command | flattened transform] into the reused scratch buffer:
// no heap allocation per step.
void ls3DRNN::prepare_mlp_input(const vector<float>& command, vector<float>& out) const
{
	out.clear();
	out.reserve(command.size() + STATE_DIM);
	out.insert(out.end(), command.begin(), command.end());
	const float* mat_ptr = glm::value_ptr(m_hidden_state);
	out.insert(out.end(), mat_ptr, mat_ptr + STATE_DIM);
}

// Public method for ls3DRNN that calls the internal MLP mutation
void ls3DRNN::mutate(float mutation_rate, float mutation_amount)
{
    m_cell->mutate(mutation_rate, mutation_amount);
}

const glm::mat4& ls3DRNN::process_step(const vector<float>& command, float max_speed)
{
	prepare_mlp_input(command, m_mlp_input);
	vector<float> params = m_cell->forward(m_mlp_input);
	glm::mat4 delta_transform = create_transform_from_params(params, max_speed);
	m_hidden_state = m_hidden_state * delta_transform;
	return m_hidden_state;
}

vector<float> ls3DRNN::get_prediction(const vector<float>& command) const
{
	prepare_mlp_input(command, m_mlp_input);
	return m_cell->forward(m_mlp_input);
}

void ls3DRNN::train_step(const vector<float>& command, const vector<float>& target_params, float rotation_weight)
{
	prepare_mlp_input(command, m_mlp_input);
	m_cell->forward(m_mlp_input);
	m_cell->backward(target_params, rotation_weight);
}

void ls3DRNN::apply_and_clear_gradients(float learning_rate, size_t batch_size)
{
	m_cell->optimizer_step(learning_rate, batch_size);
}

void ls3DRNN::set_state(const glm::mat4& new_state)
{
	m_hidden_state = new_state;
}

void ls3DRNN::reset()
{
	m_hidden_state = glm::mat4(1.0f);
}

const glm::mat4& ls3DRNN::get_state() const
{
	return m_hidden_state;
}

glm::mat4 ls3DRNN::create_transform_from_params(const vector<float>& params, float max_speed) const
{
	// params[0, 1, 2] are for translation
	glm::vec3 translation(params[0], params[1], params[2]);

	// --- MAX_SPEED Enforcement ---
	// The raw output from the MLP for translation is unconstrained and can be very large,
	// causing agents to "teleport". We must enforce the maximum speed limit here.
	float speed = glm::length(translation);
	if (speed > max_speed)
	{
		// If the desired movement exceeds max_speed, scale it down.
		// glm::normalize is safe for zero-length vectors (it returns a zero vector).
		translation = glm::normalize(translation) * max_speed;
	}

	// params[3, 4, 5, 6] are for the quaternion components (x, y, z, w)
	// Create a quaternion from the MLP's output.
	// IMPORTANT: The MLP output is unconstrained, so we must normalize it
	// to ensure it represents a valid, pure rotation.
	glm::quat rotation_quat = glm::normalize(glm::quat(params[6], params[3], params[4], params[5])); // (w, x, y, z) constructor

	// Create a translation matrix
	glm::mat4 translation_matrix = glm::translate(glm::mat4(1.0f), translation);

	// Convert the normalized quaternion to a 4x4 rotation matrix
	glm::mat4 rotation_matrix = glm::mat4_cast(rotation_quat);

	// Combine translation and rotation. Note: Order matters.
	// This order first rotates the object, then translates it in world space.
	glm::mat4 transform = translation_matrix * rotation_matrix;

	return transform;
}


vector<size_t> ls3DRNN::get_hidden_layer_dims() const
{
    const auto& all_dims = m_cell->m_layer_dims;
    if (all_dims.size() <= 2)
	{
		return {}; // No hidden layers
	}
    // Return a sub-vector excluding the first (input) and last (output) elements.
    return vector<size_t>(all_dims.begin() + 1, all_dims.end() - 1);
}
