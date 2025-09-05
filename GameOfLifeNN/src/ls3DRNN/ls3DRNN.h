#ifndef LS3DRNN_H
#define LS3DRNN_H

#define GLM_ENABLE_EXPERIMENTAL

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp> // Added for quaternion support

class ls3DRNN_MLP_Private;

using namespace std;

class ls3DRNN
{
public:
	ls3DRNN(size_t command_dim, const vector<size_t>& hidden_dims);
	
	~ls3DRNN();
	ls3DRNN(const ls3DRNN& other);
	ls3DRNN(ls3DRNN&& other) noexcept;
	ls3DRNN& operator=(const ls3DRNN& other);
	ls3DRNN& operator=(ls3DRNN&& other) noexcept;
	
	void reset();
	const glm::mat4& process_step(const vector<float>& command);
	const glm::mat4& get_state() const;
	
	// Manually sets the hidden state of the RNN. Essential for Teacher Forcing.
	// param: new_state The new 4x4 matrix to use as the hidden state.
	void set_state(const glm::mat4& new_state);
	void mutate(float mutation_rate, float mutation_amount);
	
	vector<float> get_prediction(const vector<float>& command) const;

	// This function now only calculates and accumulates gradients for one step.
	void train_step(const vector<float>& command, const vector<float>& target_params, float rotation_weight);
	// This new function applies the accumulated gradients and resets them.
	void apply_and_clear_gradients(float learning_rate, size_t batch_size);
	
	// Returns the dimensions of the hidden layers.
	vector<size_t> get_hidden_layer_dims() const;
	
private:
	glm::mat4 m_hidden_state;
	unique_ptr<ls3DRNN_MLP_Private> m_cell;
	
	// Updated: The MLP now outputs 3 translation values and 4 quaternion values.
	static constexpr size_t MLP_OUTPUT_DIM = 7;
	static constexpr size_t STATE_DIM = 16;
	
	vector<float> prepare_mlp_input(const vector<float>& command) const;
	glm::mat4 create_transform_from_params(const vector<float>& params) const;
};

#endif // LS3DRNN_H
