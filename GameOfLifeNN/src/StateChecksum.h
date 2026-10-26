#ifndef STATE_CHECKSUM_H
#define STATE_CHECKSUM_H

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

// Small non-cryptographic FNV-1a hash for deterministic replay checks. Hash
// float bits rather than formatted text so the result cannot hide drift.
namespace state_checksum
{
	using type = std::uint64_t;
	constexpr type initial = 14695981039346656037ULL;

	inline void add(type& hash, const std::uint64_t value)
	{
		for(unsigned int i = 0; i < 8; ++i)
		{
			const auto byte = static_cast<std::uint8_t>(value >> (i * 8));
			hash = (hash ^ byte) * 1099511628211ULL;
		}
	}

	inline void add(type& hash, const float value)
	{
		add(hash, static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(value)));
	}

	inline void add(type& hash, const std::span<const float> values)
	{
		add(hash, values.size());
		for(const float value : values) add(hash, value);
	}
}

#endif // STATE_CHECKSUM_H
