// Minimal framework-free unit-test harness for golnn.
// Each test file defines TEST_CASEs and ends with TEST_MAIN().
#ifndef GOLNN_TEST_FRAMEWORK_H
#define GOLNN_TEST_FRAMEWORK_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace golnn_test
{
	struct Registry
	{
		std::vector<std::pair<std::string, std::function<void()>>> cases;
		int failures = 0;
		int checks = 0;
		static Registry& instance() { static Registry r; return r; }
	};

	struct Registrar
	{
		Registrar(const char* name, std::function<void()> fn)
		{
			Registry::instance().cases.emplace_back(name, std::move(fn));
		}
	};

	inline void report(bool ok, const char* expr, const char* file, int line)
	{
		Registry& r = Registry::instance();
		++r.checks;
		if(!ok)
		{
			++r.failures;
			std::printf("  FAIL %s:%d: %s\n", file, line, expr);
		}
	}

	inline int run_all()
	{
		Registry& r = Registry::instance();
		for(auto& [name, fn] : r.cases)
		{
			const int before = r.failures;
			std::printf("CASE %s\n", name.c_str());
			fn();
			std::printf("  %s\n", r.failures == before ? "PASS" : "FAIL");
		}
		std::printf("total: %d checks, %d failures\n", r.checks, r.failures);
		return r.failures == 0 ? 0 : 1;
	}
} // namespace golnn_test

#define TEST_CASE(name) \
	static void name(); \
	static ::golnn_test::Registrar registrar_##name(#name, name); \
	static void name()

#define CHECK(cond) ::golnn_test::report((cond), #cond, __FILE__, __LINE__)

#define CHECK_NEAR(a, b, eps) \
	::golnn_test::report(std::fabs((a) - (b)) <= (eps), #a " ~= " #b, __FILE__, __LINE__)

#define TEST_MAIN int main() { return ::golnn_test::run_all(); }

#endif
