#ifndef BRAIN_CATALOG_H
#define BRAIN_CATALOG_H

#include <cstddef>
#include <string_view>
#include <vector>

// Version 1: fixed tensor shapes make agents safe to batch by tier. CUSTOM is
// retained for hand-built/test brains and always uses the scalar reference path.
enum class BrainTier : std::size_t
{
	CUSTOM = 0,
	SMALL = 1,
	MEDIUM = 2,
	LARGE = 3
};

inline constexpr std::size_t BRAIN_TIER_COUNT = 4;

inline const std::vector<std::size_t>& movement_hidden_dims(const BrainTier tier)
{
	static const std::vector<std::size_t> small{32, 32, 32};
	static const std::vector<std::size_t> medium{48, 48, 48, 48};
	static const std::vector<std::size_t> large{64, 64, 64, 64, 64};
	static const std::vector<std::size_t> custom;
	switch(tier)
	{
		case BrainTier::SMALL: return small;
		case BrainTier::MEDIUM: return medium;
		case BrainTier::LARGE: return large;
		case BrainTier::CUSTOM: return custom;
	}
	return custom;
}

inline std::size_t state_hidden_dim(const BrainTier tier)
{
	switch(tier)
	{
		case BrainTier::SMALL: return 12;
		case BrainTier::MEDIUM: return 24;
		case BrainTier::LARGE: return 32;
		case BrainTier::CUSTOM: return 0;
	}
	return 0;
}

inline std::string_view brain_tier_name(const BrainTier tier)
{
	switch(tier)
	{
		case BrainTier::SMALL: return "small";
		case BrainTier::MEDIUM: return "medium";
		case BrainTier::LARGE: return "large";
		case BrainTier::CUSTOM: return "custom";
	}
	return "custom";
}

#endif // BRAIN_CATALOG_H
