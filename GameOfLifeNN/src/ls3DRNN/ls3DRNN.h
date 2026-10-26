#ifndef LS3DRNN_H
#define LS3DRNN_H

#define GLM_ENABLE_EXPERIMENTAL

#include <vector>
#include <memory>
#include <cstdint>
#include <random>
#include <span>

#include "StateChecksum.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp> // Added for quaternion support

class ls3DRNN_MLP_Private;

using namespace std;

class ls3DRNN
{
public:
	ls3DRNN(size_t command_dim, const vector<size_t>& hidden_dims, std::uint32_t random_seed = std::random_device{}());
	ls3DRNN(size_t command_dim, const vector<size_t>& hidden_dims, std::mt19937& rng);
	
	~ls3DRNN();
	ls3DRNN(const ls3DRNN& other);
	ls3DRNN(ls3DRNN&& other) noexcept;
	ls3DRNN& operator=(const ls3DRNN& other);
	ls3DRNN& operator=(ls3DRNN&& other) noexcept;
	
	void reset();
	// The process_step function now takes a max_speed parameter to clamp the translation.
	const glm::mat4& process_step(const vector<float>& command, float max_speed);
	const glm::mat4& get_state() const;
	
	// Manually sets the hidden state of the RNN. Essential for Teacher Forcing.
	// param: new_state The new 4x4 matrix to use as the hidden state.
	void set_state(const glm::mat4& new_state);
	void mutate(float mutation_rate, float mutation_amount, std::mt19937& rng);
	
	vector<float> get_prediction(const vector<float>& command) const;

	// This function now only calculates and accumulates gradients for one step.
	void train_step(const vector<float>& command, const vector<float>& target_params, float rotation_weight);
	// All brains must have identical layer dimensions. Inputs/targets are arrays
	// of pointers to per-agent scratch, so batching adds no steady-state copies.
	static void train_batch(std::span<ls3DRNN* const> brains,
		std::span<const vector<float>* const> commands,
		std::span<const vector<float>* const> targets, float rotation_weight);
	// This new function applies the accumulated gradients and resets them.
	void apply_and_clear_gradients(float learning_rate, size_t batch_size);
	
	// Returns the dimensions of the hidden layers.
	vector<size_t> get_hidden_layer_dims() const;
	state_checksum::type checksum() const;
	
private:
	glm::mat4 m_hidden_state;
	unique_ptr<ls3DRNN_MLP_Private> m_cell;
	// Reused input assembly buffer (command + flattened transform), avoids a
	// fresh heap allocation on every process_step/get_prediction/train_step.
	mutable vector<float> m_mlp_input;
	
	// Updated: The MLP now outputs 3 translation values and 4 quaternion values.
	static constexpr size_t MLP_OUTPUT_DIM = 7;
	static constexpr size_t STATE_DIM = 16;
	
	void prepare_mlp_input(const vector<float>& command, vector<float>& out) const;
	// This function also needs the max_speed parameter.
	glm::mat4 create_transform_from_params(const vector<float>& params, float max_speed) const;
};

#endif // LS3DRNN_H
