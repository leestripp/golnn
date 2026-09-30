#include "lsVecTokenizer.h"

lsVecTokenizer::lsVecTokenizer(size_t vector_size) :
	m_current_token_id(0),
	m_vector_size(vector_size)
{}

// Add a new token with a one-hot style vector
void lsVecTokenizer::add_token_onehot(size_t token_id)
{
	if(token_id >= m_vector_size)
	{
		cerr << "ERROR: Token ID " << token_id << " is out of bounds for vector size " << m_vector_size << endl;
		return;
	}

	if(m_token_map.find(token_id) == m_token_map.end())
	{
		vector<float> v(m_vector_size, 0.0f);
		v[token_id] = 1.0f;
		m_token_map[token_id] = v;
	}
}

// Find vector for tokenId.
const vector<float>& lsVecTokenizer::find_vector(size_t token_id)
{
	auto it = m_token_map.find(token_id);
	if(it != m_token_map.end())
	{
		return it->second;
	}
	throw runtime_error("Token not found: " + to_string(token_id));
}

// Find the token ID for a given vector
size_t lsVecTokenizer::find_tokenId(const vector<float>& predicted_vector) const
{
	size_t closest_token_id = 0;
	float max_val = -1.0f;

	for(const auto& pair : m_token_map)
	{
		if (predicted_vector[pair.first] > max_val)
		{
			max_val = predicted_vector[pair.first];
			closest_token_id = pair.first;
		}
	}
	return closest_token_id;
}

// Find a list of N closest token IDs for a given vector
vector<size_t> lsVecTokenizer::find_n_closest_tokenIds(const vector<float>& predicted_vector, size_t n) const
{
	// A priority queue to keep track of the N closest tokens found so far.
	// The pair stores {distance, token_id}.
	// We use greater to make it a min-heap, so the smallest element (largest distance) is at the top.
	priority_queue<pair<float, size_t>> closest_tokens;

	for(const auto& pair : m_token_map)
	{
		size_t token_id = pair.first;
		const vector<float>& token_vector = pair.second;

		float distance = 0.0f;
		for(size_t k = 0; k < predicted_vector.size(); k++)
		{
			const float diff = predicted_vector[k] - token_vector[k];
			distance += diff * diff;
		}

		// If the priority queue has less than N elements, just push the new token.
		if(closest_tokens.size() < n)
		{
			closest_tokens.push({distance, token_id});

		} else if(distance < closest_tokens.top().first)
		{
			// If the new token is closer than the farthest token in the queue, replace it.
			closest_tokens.pop();
			closest_tokens.push({distance, token_id});
		}
	}

	// Extract the token IDs from the priority queue into a result vector.
	vector<size_t> result;
	while(! closest_tokens.empty())
	{
		// using push_back for performace.
		result.push_back(closest_tokens.top().second);
		closest_tokens.pop();
	}

	// The priority queue naturally sorts from farthest to closest (due to it being a max-heap of distances).
	// Reverse the vector to get the order from closest to farthest.
	reverse(result.begin(), result.end());

	return result;
}

// print all tokens and their vectors
void lsVecTokenizer::print() const
{
	cout << "--- Tokenizer Contents ---" << endl;
	for(const auto& pair : m_token_map)
	{
		cout << "Token ID: " << pair.first << ", Vector: ";
		print_vector(pair.second);
	}
	cout << "--------------------------" << endl;
}


// debug.
// Print a vector
void lsVecTokenizer::print_vector(const vector<float>& v) const
{
	cout << "[";
	for(size_t i = 0; i < v.size(); ++i)
	{
		cout << v[i] << (i < v.size() - 1 ? ", " : "");
	}
	cout << "]" << endl;
}

void lsVecTokenizer::print_vector(const vector<size_t>& v) const
{
	cout << "[";
	for(size_t i = 0; i < v.size(); ++i)
	{
		cout << v[i] << (i < v.size() - 1 ? ", " : "");
	}
	cout << "]" << endl;
}
