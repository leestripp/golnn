#ifndef PERCEPTION_H
#define PERCEPTION_H

#include <cstddef>

// Perception layout v2. Channels 0-30 are preserved from v1; richer motion
// and context channels are appended so older documentation remains readable.
namespace perception
{
	inline constexpr std::size_t VERSION = 2;
	inline constexpr std::size_t FOOD_DIRECTION = 0;
	inline constexpr std::size_t FOOD_DISTANCE = 3;
	inline constexpr std::size_t THREAT_DIRECTION = 4;
	inline constexpr std::size_t THREAT_DISTANCE = 7;
	inline constexpr std::size_t FRIEND_DIRECTION = 8;
	inline constexpr std::size_t FRIEND_DISTANCE = 11;
	inline constexpr std::size_t SCENT_DIRECTION = 12;
	inline constexpr std::size_t SCENT_INTENSITY = 15;
	inline constexpr std::size_t SOUND_DIRECTION = 16;
	inline constexpr std::size_t SOUND_INTENSITY = 19;
	inline constexpr std::size_t BOUNDARY_DIRECTION = 20;
	inline constexpr std::size_t FRIENDLY_DENSITY = 23;
	inline constexpr std::size_t HOSTILE_DENSITY = 24;
	inline constexpr std::size_t ENERGY = 25;
	inline constexpr std::size_t AGE = 26;
	inline constexpr std::size_t STATE_ONE_HOT = 27;
	inline constexpr std::size_t VELOCITY = 31;
	inline constexpr std::size_t HEADING = 34;
	inline constexpr std::size_t GOAL_DIRECTION = 37;
	inline constexpr std::size_t ENERGY_TREND = 40;
	inline constexpr std::size_t FOOD_DENSITY = 41;
	inline constexpr std::size_t BOUNDARY_PROXIMITY = 42;
	inline constexpr std::size_t THREAT_PROXIMITY = 43;
	inline constexpr std::size_t GOAL_ALIGNMENT = 44;
	inline constexpr std::size_t COUNT = 45;

	// The state brain keeps its original 12 one-hot bins and adds five compact
	// normalized context channels rather than copying the whole movement input.
	inline constexpr std::size_t STATE_BINNED_COUNT = 12;
	inline constexpr std::size_t STATE_ENERGY_TREND = 12;
	inline constexpr std::size_t STATE_FOOD_DENSITY = 13;
	inline constexpr std::size_t STATE_BOUNDARY_PROXIMITY = 14;
	inline constexpr std::size_t STATE_GOAL_ALIGNMENT = 15;
	inline constexpr std::size_t STATE_SPEED = 16;
	inline constexpr std::size_t STATE_INPUT_COUNT = 17;
}

#endif // PERCEPTION_H
