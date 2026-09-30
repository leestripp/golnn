// Unit tests for lsVecTokenizer: one-hot add/find round-trip, nearest-token
// lookup from a noisy vector.

#include "lsVecTokenizer.h"
#include "test_framework.h"

TEST_CASE(onehot_add_find_roundtrip)
{
	lsVecTokenizer tok(5);
	for(size_t i = 0; i < 5; ++i) tok.add_token_onehot(i);

	for(size_t i = 0; i < 5; ++i)
	{
		const vector<float>& v = tok.find_vector(i);
		CHECK(v.size() == 5);
		for(size_t j = 0; j < 5; ++j)
			CHECK(v[j] == (j == i ? 1.0f : 0.0f));
		CHECK(tok.find_tokenId(v) == i);
	}
}

TEST_CASE(closest_token_from_noisy_vector)
{
	lsVecTokenizer tok(4);
	for(size_t i = 0; i < 4; ++i) tok.add_token_onehot(i);

	// Perturb the one-hot for token 2 slightly; nearest must still be 2.
	vector<float> noisy = tok.find_vector(2);
	noisy[0] += 0.05f;
	const auto closest = tok.find_n_closest_tokenIds(noisy, 1);
	CHECK(!closest.empty());
	CHECK(closest[0] == 2);
}

TEST_MAIN
