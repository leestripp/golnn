#ifndef VULKAN_RENDERER_H
#define VULKAN_RENDERER_H

#include "RenderSnapshot.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

struct wl_display;
struct wl_surface;

class VulkanRenderer
{
public:
	VulkanRenderer();
	~VulkanRenderer();
	VulkanRenderer(const VulkanRenderer&) = delete;
	VulkanRenderer& operator=(const VulkanRenderer&) = delete;
	VulkanRenderer(VulkanRenderer&&) = delete;
	VulkanRenderer& operator=(VulkanRenderer&&) = delete;

	bool initialize(wl_display& display, wl_surface& surface,
		std::uint32_t width, std::uint32_t height, std::string& error);
	bool render(std::string& error);
	void resize(std::uint32_t width, std::uint32_t height);
	void shutdown();
	void set_render_snapshot(std::shared_ptr<const WorldSnapshot> snapshot);
	void set_grid_data(const std::vector<glm::vec3>& grid_data);
	void update_scene(bool key_w, bool key_s, bool key_a, bool key_d, bool is_sprinting);
	void mouse_motion(double x, double y);
	void mouse_button(bool pressed);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

#endif // VULKAN_RENDERER_H
