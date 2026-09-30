// Unit tests for SpatialGrid: insertion, box queries, clamping, clear/reuse.

#include "SpatialGrid.h"
#include "test_framework.h"

#include <set>

namespace
{
	std::set<size_t> query(const SpatialGrid& grid, const glm::vec3& c, float r)
	{
		std::set<size_t> out;
		grid.visit(c, r, [&](size_t i) { out.insert(i); });
		return out;
	}
}

TEST_CASE(insert_and_query_returns_index)
{
	SpatialGrid grid(100.0f, 30.0f);
	grid.insert(glm::vec3(0.0f), 7);
	grid.insert(glm::vec3(10.0f, 0.0f, 0.0f), 42);

	const auto near_origin = query(grid, glm::vec3(0.0f), 10.0f);
	CHECK(near_origin.count(7) == 1);
	CHECK(near_origin.count(42) == 1);
	CHECK(grid.entity_count() == 2);
}

TEST_CASE(query_excludes_far_cells)
{
	SpatialGrid grid(100.0f, 30.0f);
	grid.insert(glm::vec3(-49.0f, -49.0f, -49.0f), 1); // far corner cell
	grid.insert(glm::vec3(0.0f), 2);

	const auto found = query(grid, glm::vec3(0.0f), 5.0f);
	CHECK(found.count(2) == 1);
	CHECK(found.count(1) == 0); // opposite corner must not be visited
}

TEST_CASE(box_queries_are_symmetric)
{
	SpatialGrid grid(100.0f, 30.0f);
	grid.insert(glm::vec3(5.0f, 0.0f, 0.0f), 9);

	// A query from the entity's own position must find it.
	CHECK(query(grid, glm::vec3(5.0f, 0.0f, 0.0f), 0.5f).count(9) == 1);
	// And from the other side at the same radius.
	CHECK(query(grid, glm::vec3(-5.0f, 0.0f, 0.0f), 11.0f).count(9) == 1);
}

TEST_CASE(out_of_world_positions_clamp_into_edge_cells)
{
	SpatialGrid grid(100.0f, 30.0f);
	// World is [-50, 50]; these sit outside it and must not corrupt state.
	grid.insert(glm::vec3(1000.0f, 0.0f, 0.0f), 3);
	grid.insert(glm::vec3(-1000.0f, 0.0f, 0.0f), 4);

	const auto found = query(grid, glm::vec3(49.0f, 0.0f, 0.0f), 30.0f);
	CHECK(found.count(3) == 1); // clamped into +x edge cell
	const auto found_neg = query(grid, glm::vec3(-49.0f, 0.0f, 0.0f), 30.0f);
	CHECK(found_neg.count(4) == 1);
	CHECK(grid.entity_count() == 2);
}

TEST_CASE(clear_resets_but_keeps_capacity)
{
	SpatialGrid grid(100.0f, 30.0f);
	grid.insert(glm::vec3(0.0f), 1);
	grid.insert(glm::vec3(1.0f), 2);
	grid.clear();

	CHECK(grid.entity_count() == 0);
	CHECK(query(grid, glm::vec3(0.0f), 100.0f).empty());

	// Reuse after clear works.
	grid.insert(glm::vec3(0.0f), 5);
	CHECK(grid.entity_count() == 1);
	CHECK(query(grid, glm::vec3(0.0f), 5.0f).count(5) == 1);
}

TEST_MAIN
