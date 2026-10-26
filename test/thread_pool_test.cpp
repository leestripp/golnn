#include "ThreadPool.h"
#include "test_framework.h"

#include <atomic>
#include <stdexcept>
#include <vector>

TEST_CASE(parallel_for_processes_every_index_exactly_once)
{
	for(const std::size_t workers : {std::size_t(1), std::size_t(4)})
	{
		ThreadPool pool(workers);
		std::vector<std::atomic<int>> visits(1000);
		for(auto& visit : visits) visit.store(0);
		pool.parallel_for(visits.size(), 17, [&](const std::size_t begin, const std::size_t end) {
			for(std::size_t i = begin; i < end; ++i) visits[i].fetch_add(1);
		});
		for(const auto& visit : visits) CHECK(visit.load() == 1);
		pool.parallel_for(0, 1, [](std::size_t, std::size_t) { CHECK(false); });
	}
}

TEST_CASE(exception_is_rethrown_and_pool_remains_usable)
{
	ThreadPool pool(4);
	std::atomic<std::size_t> calls = 0;
	bool threw = false;
	try
	{
		pool.parallel_for(100, 3, [&](const std::size_t begin, const std::size_t end) {
			for(std::size_t i = begin; i < end; ++i)
			{
				++calls;
				if(i == 51) throw std::runtime_error("intentional test failure");
			}
		});
	}
	catch(const std::runtime_error&)
	{
		threw = true;
	}
	CHECK(threw);
	// The throwing chunk stops at item 51 (52 and 53 are not visited), but
	// every other chunk must finish and the pool must remain usable.
	CHECK(calls.load() == 98);

	std::atomic<std::size_t> second_calls = 0;
	pool.parallel_for(100, 5, [&](const std::size_t begin, const std::size_t end) {
		for(std::size_t i = begin; i < end; ++i) ++second_calls;
	});
	CHECK(second_calls.load() == 100);
}

TEST_MAIN
