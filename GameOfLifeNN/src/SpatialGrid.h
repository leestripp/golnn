#ifndef SPATIALGRID_H
#define SPATIALGRID_H

#include <glm/glm.hpp>
#include <cstddef>
#include <vector>

// Uniform spatial hash grid over the simulation world.
//
// Rebuilt once per tick, then answers "what is near (x,y,z)?" queries in
// O(k) instead of the O(n) full scans the simulation used to do for every
// agent. Cell size is chosen to cover the largest *local* query radius
// (scent/sound = 30 units); larger-radius queries visit proportionally
// more cells via visit_radius().
class SpatialGrid
{
public:
	SpatialGrid(float world_size, float cell_size);

	// Remove all entities but keep cell storage allocated (rebuilt every tick).
	void clear();

	// Insert an entity. Callers pass a stable index (e.g. position in the
	// Simulation vector) which is what queries return.
	void insert(const glm::vec3& pos, size_t index);

	// Visit every index whose cell overlaps the axis-aligned box
	// [center - radius, center + radius]. The callback may be invoked for
	// an index more than once if it sits on a cell boundary, so callers
	// that care should de-duplicate (none of our queries do: they are
	// idempotent min/max scans).
	template <typename F>
	void visit(const glm::vec3& center, float radius, F&& callback) const
	{
		visit_radius(center, radius, radius, radius, callback);
	}

	// Per-axis radii for box queries (used when the caller wants a
	// radius that differs per axis).
	template <typename F>
	void visit_radius(const glm::vec3& center, float rx, float ry, float rz, F&& callback) const
	{
		if (m_cells.empty()) return;

		const int x0 = cell_index_x(center.x - rx);
		const int x1 = cell_index_x(center.x + rx);
		const int y0 = cell_index_y(center.y - ry);
		const int y1 = cell_index_y(center.y + ry);
		const int z0 = cell_index_z(center.z - rz);
		const int z1 = cell_index_z(center.z + rz);

		for (int z = z0; z <= z1; ++z)
		for (int y = y0; y <= y1; ++y)
		for (int x = x0; x <= x1; ++x)
		{
			const std::vector<size_t>& cell = m_cells[cell_offset(x, y, z)];
			for (size_t idx : cell)
			{
				callback(idx);
			}
		}
	}

	size_t entity_count() const { return m_count; }

private:
	int cell_index_x(float v) const { return clamp_index(static_cast<int>((v - m_min.x) * m_inv_cell)); }
	int cell_index_y(float v) const { return clamp_index(static_cast<int>((v - m_min.y) * m_inv_cell)); }
	int cell_index_z(float v) const { return clamp_index(static_cast<int>((v - m_min.z) * m_inv_cell)); }
	int clamp_index(int i) const { return i < 0 ? 0 : (i >= m_dim ? m_dim - 1 : i); }
	size_t cell_offset(int x, int y, int z) const
	{
		return (static_cast<size_t>(z) * static_cast<size_t>(m_dim) + static_cast<size_t>(y)) * static_cast<size_t>(m_dim) + static_cast<size_t>(x);
	}

	float m_cell_size;
	float m_inv_cell;
	int m_dim;
	glm::vec3 m_min;
	std::vector<std::vector<size_t>> m_cells;
	size_t m_count = 0;
};

#endif // SPATIALGRID_H
