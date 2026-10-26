#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

// Persistent indexed work pool. The calling thread executes chunks too, so a
// pool size of N means N total participants rather than N plus the caller.
class ThreadPool
{
public:
	explicit ThreadPool(std::size_t worker_count);
	~ThreadPool();

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	std::size_t size() const { return m_worker_count; }
	void parallel_for(std::size_t count, std::size_t grain_size,
	                  const std::function<void(std::size_t, std::size_t)>& work);

private:
	void worker_loop(std::stop_token stop_token);
	bool claim_chunk(std::size_t& begin, std::size_t& end);
	void finish_chunk();

	std::size_t m_worker_count;
	std::vector<std::jthread> m_threads;

	std::mutex m_mutex;
	std::condition_variable m_work_cv;
	std::condition_variable m_done_cv;
	std::size_t m_generation = 0;
	bool m_stopping = false;

	const std::function<void(std::size_t, std::size_t)>* m_work = nullptr;
	std::size_t m_count = 0;
	std::size_t m_next = 0;
	std::size_t m_grain_size = 1;
	std::size_t m_remaining = 0;
	std::exception_ptr m_error;
};

#endif // THREAD_POOL_H
