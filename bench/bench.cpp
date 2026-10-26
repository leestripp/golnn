// golnn headless benchmark.
// Runs Simulation::tick() without GTK/GL and reports timing and a replay hash.
// Positional usage: golnn_bench [ticks] [agents] [food] [world_size]
// Options: --samples N (fresh runs, default 1), --seed N (default 12345)

#include "Simulation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
	struct Options
	{
		std::size_t ticks = 1000;
		std::size_t agents = 100;
		std::size_t food = 50;
		float world = 100.0f;
		std::size_t samples = 1;
		std::uint32_t seed = 12345;
		std::size_t workers = 0;
		bool fixed_food = false;
		std::string telemetry_path;
		BrainTier brain_tier = BrainTier::SMALL;
		bool use_batched_brains = true;
	};

	std::size_t parse_size(const std::string_view value, const char* name)
	{
		std::size_t consumed = 0;
		const unsigned long parsed = std::stoul(std::string(value), &consumed);
		if(consumed != value.size() || parsed == 0)
			throw std::invalid_argument(std::string("invalid ") + name + ": " + std::string(value));
		return parsed;
	}

	float parse_world(const std::string_view value)
	{
		std::size_t consumed = 0;
		const float parsed = std::stof(std::string(value), &consumed);
		if(consumed != value.size() || parsed <= 0.0f)
			throw std::invalid_argument("world size must be positive");
		return parsed;
	}

	Options parse_options(int argc, char** argv)
	{
		Options options;
		std::vector<std::string_view> positional;
		for(int i = 1; i < argc; ++i)
		{
			const std::string_view arg(argv[i]);
			if(arg == "--help" || arg == "-h")
			{
				std::cout << "usage: golnn_bench [ticks] [agents] [food] [world_size] "
				             "[--samples N] [--seed N] [--workers N] [--fixed-food] [--telemetry PATH] [--brain-tier small|medium|large] [--scalar-brains]\n";
				std::exit(0);
			}
			if(arg == "--scalar-brains")
			{
				options.use_batched_brains = false;
			}
			else if(arg == "--brain-tier")
			{
				if(++i >= argc) throw std::invalid_argument("--brain-tier requires a value");
				const std::string_view value(argv[i]);
				if(value == "small") options.brain_tier = BrainTier::SMALL;
				else if(value == "medium") options.brain_tier = BrainTier::MEDIUM;
				else if(value == "large") options.brain_tier = BrainTier::LARGE;
				else throw std::invalid_argument("unknown brain tier: " + std::string(value));
			}
			else if(arg == "--fixed-food")
			{
				options.fixed_food = true;
			}
			else if(arg == "--telemetry")
			{
				if(++i >= argc) throw std::invalid_argument("--telemetry requires a path");
				options.telemetry_path = argv[i];
			}
			else if(arg == "--samples" || arg == "--seed" || arg == "--workers")
			{
				if(++i >= argc) throw std::invalid_argument(std::string(arg) + " requires a value");
				const std::string_view value(argv[i]);
				if(arg == "--samples") options.samples = parse_size(value, "sample count");
				else if(arg == "--workers") options.workers = parse_size(value, "worker count");
				else options.seed = static_cast<std::uint32_t>(parse_size(value, "seed"));
			}
			else if(!arg.empty() && arg[0] == '-')
				throw std::invalid_argument("unknown option: " + std::string(arg));
			else
				positional.push_back(arg);
		}
		if(positional.size() > 4) throw std::invalid_argument("too many positional arguments");
		if(!positional.empty()) options.ticks = parse_size(positional[0], "tick count");
		if(positional.size() > 1) options.agents = parse_size(positional[1], "agent count");
		if(positional.size() > 2) options.food = parse_size(positional[2], "food count");
		if(positional.size() > 3) options.world = parse_world(positional[3]);
		return options;
	}

	double run_sample(const Options& options, const std::size_t sample,
	                  state_checksum::type& checksum, std::size_t& final_agents,
	                  std::size_t& final_food)
	{
		SimulationConfig config;
		config.seed = options.seed;
		config.worker_count = options.workers;
		config.brain_tier = options.brain_tier;
		config.use_batched_brains = options.use_batched_brains;
		config.publish_render_snapshots = false;
		config.food_regrowth_mode = options.fixed_food ? FoodRegrowthMode::FIXED : FoodRegrowthMode::ADAPTIVE;
		config.telemetry_path = options.telemetry_path;
		Simulation sim(options.agents, options.food, options.world, config);
		sim.tick(); // Warm caches; every sample starts from the same seeded state.

		const auto t0 = std::chrono::steady_clock::now();
		for(std::size_t i = 0; i < options.ticks; ++i) sim.tick();
		const auto t1 = std::chrono::steady_clock::now();

		const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		const EcologyStats stats = sim.ecology_stats();
		checksum = sim.state_checksum();
		final_agents = sim.m_agents.size();
		final_food = sim.m_food.size();
		const double ms_per_tick = total_ms / static_cast<double>(options.ticks);
		std::cout << "workers=" << sim.worker_count() << " sample=" << sample << " total_ms=" << total_ms
		          << " ms_per_tick=" << ms_per_tick
		          << " ticks_per_sec=" << 1000.0 / ms_per_tick
		          << " final_agents=" << final_agents
		          << " final_food=" << final_food
		          << " feeders=" << stats.feeders
		          << " predators=" << stats.predators
		          << " vegetation=" << stats.vegetables
		          << " births=" << stats.total_births
		          << " deaths=" << stats.total_deaths
		          << " vegetation_eaten=" << stats.total_vegetables_eaten
		          << " autonomy=" << stats.mean_autonomy
		          << " rest=" << stats.state_counts[4]
		          << " graze=" << stats.state_counts[5]
		          << " explore=" << stats.state_counts[6]
		          << " investigate=" << stats.state_counts[7]
		          << " checksum=0x" << std::hex << checksum << std::dec << '\n';
		return ms_per_tick;
	}
}

int main(int argc, char** argv)
{
	try
	{
		const Options options = parse_options(argc, argv);
		std::cout << "bench: ticks=" << options.ticks
		          << " agents=" << options.agents
		          << " food=" << options.food
		          << " world=" << options.world
		          << " samples=" << options.samples
		          << " seed=" << options.seed
		          << " brain_tier=" << brain_tier_name(options.brain_tier)
		          << " batched=" << (options.use_batched_brains ? "yes" : "no") << '\n';

		std::vector<double> times;
		times.reserve(options.samples);
		state_checksum::type first_checksum = 0;
		bool checksums_match = true;
		for(std::size_t sample = 0; sample < options.samples; ++sample)
		{
			state_checksum::type checksum = 0;
			std::size_t final_agents = 0;
			std::size_t final_food = 0;
			times.push_back(run_sample(options, sample, checksum, final_agents, final_food));
			if(sample == 0) first_checksum = checksum;
			else checksums_match = checksums_match && checksum == first_checksum;
		}

		std::sort(times.begin(), times.end());
		const double median = times.size() % 2 == 0
			? (times[times.size() / 2 - 1] + times[times.size() / 2]) / 2.0
			: times[times.size() / 2];
		std::cout << std::fixed << std::setprecision(4)
		          << "result: median_ms_per_tick=" << median
		          << " min_ms_per_tick=" << times.front()
		          << " max_ms_per_tick=" << times.back()
		          << " median_ticks_per_sec=" << 1000.0 / median
		          << " reproducible=" << (checksums_match ? "yes" : "NO") << '\n';
		return checksums_match ? 0 : 2;
	}
	catch(const std::exception& error)
	{
		std::cerr << "golnn_bench: " << error.what() << '\n';
		return 1;
	}
}
