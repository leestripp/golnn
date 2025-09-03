#include "lsRNN.h"

// Utils.

pair<double, double> lsRNN::find_min_max(const vector<double>& data)
{
	double min_val = numeric_limits<double>::max();
	double max_val = numeric_limits<double>::min();
	for(double val : data)
	{
		if(val < min_val) min_val = val;
		if(val > max_val) max_val = val;
	}
	return {min_val, max_val};
}

// Normalize a vector using Min-Max scaling
vector<double> lsRNN::normalize_data(const vector<double>& data, double min_val, double max_val)
{
	vector<double> normalized_data(data.size());
	double range = max_val - min_val;
	
	// Prevent division by zero.
	if(range < 1e-9)
	{
		return vector<double>(data.size(), 0.0);
	}
	
	for(size_t i = 0; i < data.size(); i++)
	{
		normalized_data[i] = (data[i] - min_val) / range;
	}
	return normalized_data;
}

// Denormalize a value back to its original scale
double lsRNN::denormalize_value(double normalized_val, double min_val, double max_val)
{
	double range = max_val - min_val;
	return normalized_val * range + min_val;
}

// definitions
double lsRNN::random_double(double min, double max)
{
	static random_device rd;
	static mt19937 gen(rd());
	uniform_real_distribution<> dis(min, max);
	return dis(gen);
}

vector<double> lsRNN::create_random_vector(size_t size)
{
	vector<double> vec(size);
	for(size_t i = 0; i < size; ++i)
	{
		vec[i] = random_double(-1.0, 1.0);
	}
	return vec;
}

void lsRNN::softmax(vector<double>& vec)
{
	double max_val = -numeric_limits<double>::infinity();
	for (double val : vec)
	{
		if (val > max_val)
		{
			max_val = val;
		}
	}

	double sum = 0.0;
	for (size_t i = 0; i < vec.size(); ++i)
	{
		vec[i] = exp(vec[i] - max_val);
		sum += vec[i];
	}
	
	for (double& val : vec)
	{
		val /= sum;
	}
}

void lsRNN::normalize(vector<double>& vec)
{
	double magnitude = 0.0;
	for(double val : vec)
	{
		magnitude += val * val;
	}
	magnitude = sqrt(magnitude);
	
	if(magnitude > 1e-6)
	{
		for(double& val : vec)
		{
			val /= magnitude;
		}
	}
}






// lsRNN Constructor
lsRNN::lsRNN(size_t input_dim, size_t hidden_dim, size_t output_dim, Task task) :
	input_dim(input_dim),
	hidden_dim(hidden_dim),
	output_dim(output_dim),
	m_task(task)
{
	// Initialize weights and biases with small random values
	W_xh = vector<vector<double>>(hidden_dim, vector<double>(input_dim));
	W_hh = vector<vector<double>>(hidden_dim, vector<double>(hidden_dim));
	W_ho = vector<vector<double>>(output_dim, vector<double>(hidden_dim));
	b_h = vector<double>(hidden_dim, 0.0);
	b_o = vector<double>(output_dim, 0.0);
	
	for (auto& row : W_xh) { for (double& val : row) val = random_double(-0.5, 0.5); }
	for (auto& row : W_hh) { for (double& val : row) val = random_double(-0.5, 0.5); }
	for (auto& row : W_ho) { for (double& val : row) val = random_double(-0.5, 0.5); }
	
	h = vector<double>(hidden_dim, 0.0);
	m_h_history.push_back(h);
}

// Loss functions are now member functions
double lsRNN::get_loss_cross_entropy(const vector<double>& predicted, const vector<double>& expected)
{
	double loss = 0.0;
	for (size_t i = 0; i < predicted.size(); ++i)
	{
		if (expected[i] == 1.0)
		{
			loss -= log(predicted[i] + 1e-9); // Add a small value to prevent log(0)
		}
	}
	return loss;
}

double lsRNN::get_loss_mse(const vector<double>& predicted, const vector<double>& expected)
{
	double loss = 0.0;
	for (size_t i = 0; i < predicted.size(); ++i)
	{
		loss += pow(predicted[i] - expected[i], 2);
	}
	return loss / predicted.size();
}







void lsRNN::reset_state()
{
	h = vector<double>(hidden_dim, 0.0);
	m_h_history.clear();
	m_h_history.push_back(h);
}

void lsRNN::print_weights()
{
	cout << "### Hidden Layers:" << endl;
	for(const auto& row : W_hh) {
		cout << "[";
		for(size_t i = 0; i < row.size(); ++i) {
			cout << fixed << setprecision(4) << row[i] << (i < row.size() - 1 ? ", " : "");
		}
		cout << "]" << endl;
	}
}

vector<double> lsRNN::forward_pass(const vector<double>& input_vector)
{
	vector<double> h_new = b_h;
	
	for (size_t i = 0; i < hidden_dim; ++i) {
		for (size_t j = 0; j < input_dim; ++j) {
			h_new[i] += W_xh[i][j] * input_vector[j];
		}
		for (size_t j = 0; j < hidden_dim; ++j) {
			h_new[i] += W_hh[i][j] * h[j];
		}
	}
	
	for (double& val : h_new) {
		val = tanh(val);
	}
	
	h = h_new;
	m_h_history.push_back(h_new);
	
	vector<double> output = b_o;
	for (size_t i = 0; i < output_dim; ++i) {
		for (size_t j = 0; j < hidden_dim; ++j) {
			output[i] += W_ho[i][j] * h_new[j];
		}
	}
	
	// Apply activation function based on the task
	if (m_task == Task::CLASSIFICATION) {
		softmax(output);
	}
	// For regression, we use a linear output, so no activation is applied.
	
	return output;
}

void lsRNN::backpropagate(
	const vector<vector<double>>& inputs,
	const vector<vector<double>>& outputs,
	const vector<vector<double>>& expected_outputs,
	double learning_rate)
{
	vector<vector<double>> dW_xh(hidden_dim, vector<double>(input_dim, 0.0));
	vector<vector<double>> dW_hh(hidden_dim, vector<double>(hidden_dim, 0.0));
	vector<vector<double>> dW_ho(output_dim, vector<double>(hidden_dim, 0.0));
	vector<double> db_h(hidden_dim, 0.0);
	vector<double> db_o(output_dim, 0.0);
	
	vector<double> d_h_next(hidden_dim, 0.0);
	
	for(int t = inputs.size() - 1; t >= 0; --t)
	{
		vector<double> d_output(output_dim);
		
		// Calculate the output gradient based on the task
		switch (m_task) {
			case Task::CLASSIFICATION:
				// For Softmax + Cross-Entropy Loss, the output gradient is simply predicted - expected
				for(size_t i = 0; i < output_dim; ++i) {
					d_output[i] = outputs[t][i] - expected_outputs[t][i];
				}
				break;
			case Task::REGRESSION:
				// For MSE loss, the output gradient is 2 * (predicted - expected)
				for(size_t i = 0; i < output_dim; ++i) {
					d_output[i] = 2 * (outputs[t][i] - expected_outputs[t][i]) / output_dim;
				}
				break;
		}
		
		for(size_t i = 0; i < output_dim; ++i)
		{
			db_o[i] += d_output[i];
			for(size_t j = 0; j < hidden_dim; ++j)
			{
				dW_ho[i][j] += d_output[i] * m_h_history[t+1][j];
			}
		}
		
		vector<double> d_h(hidden_dim, 0.0);
		for(size_t i = 0; i < hidden_dim; ++i)
		{
			for(size_t j = 0; j < output_dim; ++j)
			{
				d_h[i] += d_output[j] * W_ho[j][i];
			}
			for(size_t j = 0; j < hidden_dim; ++j)
			{
				d_h[i] += d_h_next[j] * W_hh[j][i];
			}
		}
		
		vector<double> d_h_raw(hidden_dim);
		for(size_t i = 0; i < hidden_dim; ++i)
		{
			d_h_raw[i] = d_h[i] * (1 - m_h_history[t+1][i] * m_h_history[t+1][i]);
		}
		
		for(size_t i = 0; i < hidden_dim; ++i)
		{
			db_h[i] += d_h_raw[i];
			for(size_t j = 0; j < input_dim; ++j)
			{
				dW_xh[i][j] += d_h_raw[i] * inputs[t][j];
			}
			for(size_t j = 0; j < hidden_dim; ++j)
			{
				dW_hh[i][j] += d_h_raw[i] * m_h_history[t][j];
			}
		}
		
		d_h_next = d_h_raw;
	}
	
	double clip_value = 5.0;
	for (auto& row : dW_xh) for(double& val : row) val = std::min(std::max(val, -clip_value), clip_value);
	for (auto& row : dW_hh) for(double& val : row) val = std::min(std::max(val, -clip_value), clip_value);
	for (auto& row : dW_ho) for(double& val : row) val = std::min(std::max(val, -clip_value), clip_value);
	for (double& val : db_h) val = std::min(std::max(val, -clip_value), clip_value);
	for (double& val : db_o) val = std::min(std::max(val, -clip_value), clip_value);
	
	for(size_t i = 0; i < hidden_dim; ++i)
	{
		for(size_t j = 0; j < input_dim; ++j) W_xh[i][j] -= learning_rate * dW_xh[i][j];
		for(size_t j = 0; j < hidden_dim; ++j) W_hh[i][j] -= learning_rate * dW_hh[i][j];
		b_h[i] -= learning_rate * db_h[i];
	}
	
	for(size_t i = 0; i < output_dim; ++i)
	{
		for(size_t j = 0; j < hidden_dim; ++j) W_ho[i][j] -= learning_rate * dW_ho[i][j];
		b_o[i] -= learning_rate * db_o[i];
	}
}

