#include "ls3DRNN.h"

#include <stdexcept>
#include <numeric>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

// Internal MLP Cell
class ls3DRNN_MLP_Private
{
public:
	ls3DRNN_MLP_Private(size_t input_dim, size_t output_dim, const vector<size_t>& hidden_dims)
	{
		m_layer_dims.push_back(input_dim);
		m_layer_dims.insert(m_layer_dims.end(), hidden_dims.begin(), hidden_dims.end());
		m_layer_dims.push_back(output_dim);

		srand(static_cast<unsigned int>(time(0)));
		for(size_t i = 0; i < m_layer_dims.size() - 1; i++)
		{
			size_t in_features = m_layer_dims[i];
			size_t out_features = m_layer_dims[i + 1];

			vector<vector<float>> layer_weights(out_features, vector<float>(in_features));
			for(auto& row : layer_weights)
			{
				for(float& val : row)
				{
					val = static_cast<float>(rand()) / RAND_MAX * 0.2f - 0.1f;
				}
			}
			m_weights.push_back(move(layer_weights));
			m_biases.push_back(vector<float>(out_features, 0.0f));
			m_weight_gradients.push_back(vector<vector<float>>(out_features, vector<float>(in_features, 0.0f)));
			m_bias_gradients.push_back(vector<float>(out_features, 0.0f));
		}
	}

	vector<size_t> m_layer_dims;
	vector<vector<vector<float>>> m_weights;
	vector<vector<float>> m_biases;
	mutable vector<vector<float>> m_last_activations;
	mutable vector<vector<vector<float>>> m_weight_gradients;
	mutable vector<vector<float>> m_bias_gradients;

	vector<float> forward(const vector<float>& input) const
	{
		m_last_activations.clear();
		m_last_activations.push_back(input);

		vector<float> current_activation = input;
		for(size_t i = 0; i < m_weights.size(); i++)
		{
			vector<float> z(m_biases[i].size(), 0.0f);
			for(size_t j = 0; j < m_weights[i].size(); j++)
			{
				z[j] = inner_product(m_weights[i][j].begin(), m_weights[i][j].end(), current_activation.begin(), 0.0f) + m_biases[i][j];
			}

			current_activation = z;
			if(i < m_weights.size() - 1)
			{
				transform(current_activation.begin(), current_activation.end(), current_activation.begin(), [](float v){ return tanh(v); });
			}
			m_last_activations.push_back(current_activation);
		}
		return current_activation;
	}

	void backward(const vector<float>& target_output, float rotation_weight) const
	{
		// Updated: The target_output vector is now size 7
		vector<float> delta_a(target_output.size());
		// First 3 are for translation
		for(size_t i = 0; i < 3; ++i) { delta_a[i] = m_last_activations.back()[i] - target_output[i]; }
		// Last 4 are for the quaternion rotation, apply the weight here
		for(size_t i = 3; i < 7; ++i) { delta_a[i] = rotation_weight * (m_last_activations.back()[i] - target_output[i]); }

		for(int i = m_weights.size() - 1; i >= 0; i--)
		{
			const auto& current_activations = m_last_activations[i + 1];

			vector<float> delta_z = delta_a;
			// Apply tanh derivative only for hidden layers
			if(i < static_cast<int>(m_weights.size()) - 1)
			{
				for(size_t j = 0; j < delta_z.size(); j++)
				{
					float tanh_derivative = 1.0f - pow(current_activations[j], 2);
					delta_z[j] *= tanh_derivative;
				}
			}

			const auto& prev_activations = m_last_activations[i];

			for(size_t j = 0; j < m_weights[i].size(); ++j)
			{
				m_bias_gradients[i][j] = delta_z[j];
				for(size_t k = 0; k < m_weights[i][j].size(); ++k)
				{
					m_weight_gradients[i][j][k] = delta_z[j] * prev_activations[k];
				}
			}

			if(i > 0)
			{
				vector<float> next_delta_a(m_layer_dims[i], 0.0f);
				for (size_t j = 0; j < m_weights[i][0].size(); ++j)
				{
					float error_sum = 0.0f;
					for(size_t k = 0; k < m_weights[i].size(); ++k)
					{
						error_sum += m_weights[i][k][j] * delta_z[k];
					}
					next_delta_a[j] = error_sum;
				}
				delta_a = next_delta_a;
			}
		}
	}

	void optimizer_step(float learning_rate)
	{
		for(size_t i = 0; i < m_weights.size(); ++i)
		{
			for(size_t j = 0; j < m_weights[i].size(); ++j)
			{
				m_biases[i][j] -= learning_rate * m_bias_gradients[i][j];
				for(size_t k = 0; k < m_weights[i][j].size(); ++k)
				{
					m_weights[i][j][k] -= learning_rate * m_weight_gradients[i][j][k];
				}
			}
		}
	}
	
	// Adjusts weights and biases by a random amount.
	void mutate(float mutation_rate, float mutation_amount)
	{
		// Helper lambda to generate a random value in [-amount, +amount]
		auto random_mutation = [=]() {
			return (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * mutation_amount;
		};

		// Iterate through all layers
		for (size_t i = 0; i < m_weights.size(); ++i) {
			// Mutate weights
			for (auto& row : m_weights[i]) {
				for (float& weight : row) {
					// Check if this specific weight should be mutated
					if (static_cast<float>(rand()) / RAND_MAX < mutation_rate) {
						weight += random_mutation();
					}
				}
			}

			// Mutate biases
			for (float& bias : m_biases[i]) {
				// Check if this specific bias should be mutated
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

vector<float> ls3DRNN::prepare_mlp_input(const vector<float>& command) const
{
	vector<float> mlp_input;
	mlp_input.reserve(command.size() + STATE_DIM);
	mlp_input.insert(mlp_input.end(), command.begin(), command.end());
	const float* mat_ptr = glm::value_ptr(m_hidden_state);
	mlp_input.insert(mlp_input.end(), mat_ptr, mat_ptr + STATE_DIM);
	return mlp_input;
}

// Public method for ls3DRNN that calls the internal MLP mutation
void ls3DRNN::mutate(float mutation_rate, float mutation_amount)
{
    m_cell->mutate(mutation_rate, mutation_amount);
}

const glm::mat4& ls3DRNN::process_step(const vector<float>& command)
{
	vector<float> mlp_input = prepare_mlp_input(command);
	vector<float> params = m_cell->forward(mlp_input);
	glm::mat4 delta_transform = create_transform_from_params(params);
	m_hidden_state = m_hidden_state * delta_transform;
	return m_hidden_state;
}

vector<float> ls3DRNN::get_prediction(const vector<float>& command) const
{
	vector<float> mlp_input = prepare_mlp_input(command);
	return m_cell->forward(mlp_input);
}

void ls3DRNN::train_step(const vector<float>& command, const vector<float>& target_params, float learning_rate, float rotation_weight)
{
	vector<float> mlp_input = prepare_mlp_input(command);
	m_cell->forward(mlp_input);
	m_cell->backward(target_params, rotation_weight);
	m_cell->optimizer_step(learning_rate);
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

glm::mat4 ls3DRNN::create_transform_from_params(const vector<float>& params) const
{
	// params[0, 1, 2] are for translation
	glm::vec3 translation(params[0], params[1], params[2]);

	// params[3, 4, 5, 6] are for the quaternion components (x, y, z, w)
	// Create a quaternion from the MLP's output.
	// IMPORTANT: The MLP output is unconstrained, so we must normalize it
	// to ensure it represents a valid, pure rotation.
	glm::quat rotation_quat = glm::normalize(glm::quat(params[6], params[3], params[4], params[5])); // (w, x, y, z) constructor

	// Create a translation matrix
	glm::mat4 translation_matrix = glm::translate(glm::mat4(1.0f), translation);

	// Convert the normalized quaternion to a 4x4 rotation matrix
	glm::mat4 rotation_matrix = glm::mat4_cast(rotation_quat); // <-- THE FIX IS HERE

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

