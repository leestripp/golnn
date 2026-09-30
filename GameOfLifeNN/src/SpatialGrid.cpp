#include "SpatialGrid.h"

#include <cmath>

SpatialGrid::SpatialGrid(float world_size, float cell_size)
{
	// World is centered on the origin: [-half, +half] per axis.
	const float half = world_size / 2.0f;
	m_min = glm::vec3(-half);
	m_cell_size = (cell_size > 0.0f) ? cell_size : 1.0f;
	m_inv_cell = 1.0f / m_cell_size;

	// Enough cells to cover the world, at least 1 per axis.
	m_dim = static_cast<int>(std::ceil(world_size / m_cell_size));
	if (m_dim < 1) m_dim = 1;

	m_cells.resize(static_cast<size_t>(m_dim) * static_cast<size_t>(m_dim) * static_cast<size_t>(m_dim));
}

void SpatialGrid::clear()
{
	for (auto& cell : m_cells)
	{
		cell.clear();
	}
	m_count = 0;
}

void SpatialGrid::insert(const glm::vec3& pos, size_t index)
{
	const int x = cell_index_x(pos.x);
	const int y = cell_index_y(pos.y);
	const int z = cell_index_z(pos.z);
	m_cells[cell_offset(x, y, z)].push_back(index);
	++m_count;
}
