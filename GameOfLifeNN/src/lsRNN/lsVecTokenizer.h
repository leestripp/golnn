#ifndef LSVECTOKENIZER_H
#define LSVECTOKENIZER_H

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

// One-Hot Tokenizer.

class lsVecTokenizer
{
public:
	explicit lsVecTokenizer(size_t vector_size);

	// Add tokens.
	void add_token_onehot(size_t token_id);

	// Find.
	const vector<float>& find_vector(size_t token_id);
	size_t find_tokenId(const vector<float>& predicted_vector) const;
	vector<size_t> find_n_closest_tokenIds(const vector<float>& predicted_vector, size_t n) const;

	// print all tokens and their vectors
	void print() const;
	void print_vector(const vector<float>& v) const;
	void print_vector(const vector<size_t>& v) const;

private:
	unordered_map<size_t, vector<float>> m_token_map;
	size_t m_current_token_id;
	size_t m_vector_size;
};

#endif // LSVECTOKENIZER_H
