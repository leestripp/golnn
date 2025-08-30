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

using namespace std;

// Enum to define the task for the RNN (Classification or Regression)
enum class Task {
	CLASSIFICATION,
	REGRESSION
};


class lsRNN
{
public:
	explicit lsRNN(size_t input_dim, size_t hidden_dim, size_t output_dim, Task task);
	
	vector<double> forward_pass(const vector<double>& input_vector);
	void backpropagate(
		const vector<vector<double>>& inputs,
		const vector<vector<double>>& outputs,
		const vector<vector<double>>& expected_outputs,
		double learning_rate
	);
	
	// Loss functions are now part of the class
	double get_loss_cross_entropy(const vector<double>& predicted, const vector<double>& expected);
	double get_loss_mse(const vector<double>& predicted, const vector<double>& expected);
	
	void reset_state();
	
	// Utils.
	pair<double, double> find_min_max(const vector<double>& data);
	vector<double> normalize_data(const vector<double>& data, double min_val, double max_val);
	double denormalize_value(double normalized_val, double min_val, double max_val);
	
	// debug
	void print_nodes();
	
private:
	// Utility functions
	double random_double(double min, double max);
	vector<double> create_random_vector(size_t size);
	
	// Activations
	void softmax(vector<double>& vec);
	void normalize(vector<double>& vec);
	
	// Dimensions
	size_t input_dim;
	size_t hidden_dim;
	size_t output_dim;
	Task m_task; // Private member to store the task type
	
	// Weights and biases
	vector<vector<double>> W_hh;
	vector<vector<double>> W_xh;
	vector<vector<double>> W_ho;
	
	vector<double> b_h;
	vector<double> b_o;
	
	// State and history
	vector<double> h;
	vector<vector<double>> m_h_history;
};

#endif // LSRNN_H
