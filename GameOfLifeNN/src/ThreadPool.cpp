#include "ThreadPool.h"

#include <algorithm>
#include <stdexcept>

ThreadPool::ThreadPool(const std::size_t worker_count)
	: m_worker_count(std::max<std::size_t>(1, worker_count))
{
	try
	{
		m_threads.reserve(m_worker_count - 1);
		for(std::size_t i = 1; i < m_worker_count; ++i)
		{
			m_threads.emplace_back([this](const std::stop_token stop_token) {
				worker_loop(stop_token);
			});
		}
	}
	catch(...)
	{
		// If thread creation fails part-way through, wake existing workers
		// before joining them; otherwise destruction could wait forever.
		{
			std::lock_guard lock(m_mutex);
			m_stopping = true;
			++m_generation;
		}
		m_work_cv.notify_all();
		m_threads.clear();
		throw;
	}
}

ThreadPool::~ThreadPool()
{
	{
		std::lock_guard lock(m_mutex);
		m_stopping = true;
		++m_generation;
	}
	m_work_cv.notify_all();
	m_threads.clear();
}

void ThreadPool::parallel_for(const std::size_t count, const std::size_t grain_size,
                              const std::function<void(std::size_t, std::size_t)>& work)
{
	if(count == 0) return;
	if(grain_size == 0) throw std::invalid_argument("ThreadPool grain size must be positive");

	if(m_worker_count == 1)
	{
		work(0, count);
		return;
	}

	{
		std::lock_guard lock(m_mutex);
		m_work = &work;
		m_count = count;
		m_next = 0;
		m_grain_size = grain_size;
		m_remaining = (count + grain_size - 1) / grain_size;
		m_error = nullptr;
		++m_generation;
	}
	m_work_cv.notify_all();

	// The caller helps rather than sleeping while all useful work is done by
	// the workers. This also reduces latency for small batches.
	while(true)
	{
		std::size_t begin = 0;
		std::size_t end = 0;
		if(!claim_chunk(begin, end)) break;
		try
		{
			work(begin, end);
		}
		catch(...)
		{
			std::lock_guard lock(m_mutex);
			if(!m_error) m_error = std::current_exception();
		}
		finish_chunk();
	}

	std::exception_ptr error;
	{
		std::unique_lock lock(m_mutex);
		m_done_cv.wait(lock, [this] { return m_remaining == 0; });
		error = m_error;
		m_work = nullptr;
	}
	if(error) std::rethrow_exception(error);
}

void ThreadPool::worker_loop(const std::stop_token stop_token)
{
	std::size_t seen_generation = 0;
	while(true)
	{
		{
			std::unique_lock lock(m_mutex);
			m_work_cv.wait(lock, [&] {
				return stop_token.stop_requested() || m_stopping || m_generation != seen_generation;
			});
			if(stop_token.stop_requested() || m_stopping) return;
			seen_generation = m_generation;
		}

		while(true)
		{
			std::size_t begin = 0;
			std::size_t end = 0;
			if(!claim_chunk(begin, end)) break;
			try
			{
				(*m_work)(begin, end);
			}
			catch(...)
			{
				std::lock_guard lock(m_mutex);
				if(!m_error) m_error = std::current_exception();
			}
			finish_chunk();
		}
	}
}

bool ThreadPool::claim_chunk(std::size_t& begin, std::size_t& end)
{
	std::lock_guard lock(m_mutex);
	if(m_next >= m_count) return false;
	begin = m_next;
	end = std::min(m_count, begin + m_grain_size);
	m_next = end;
	return true;
}

void ThreadPool::finish_chunk()
{
	std::lock_guard lock(m_mutex);
	if(--m_remaining == 0) m_done_cv.notify_one();
}
