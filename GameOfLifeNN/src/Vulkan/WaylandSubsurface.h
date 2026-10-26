#ifndef WAYLAND_SUBSURFACE_H
#define WAYLAND_SUBSURFACE_H

#include <memory>
#include <string>

struct wl_compositor;
struct wl_display;
struct wl_surface;

// Owns the independent native surface used for Vulkan presentation. GTK keeps
// its top-level surface for controls; Vulkan presents into this child area.
class WaylandSubsurface
{
public:
	WaylandSubsurface();
	~WaylandSubsurface();
	WaylandSubsurface(const WaylandSubsurface&) = delete;
	WaylandSubsurface& operator=(const WaylandSubsurface&) = delete;

	bool initialize(wl_display& display, wl_compositor& compositor,
		wl_surface& parent, std::string& error);
	void set_position(int x, int y);
	void set_logical_size(int width, int height);
	wl_surface& surface() const;
	void shutdown();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

#endif // WAYLAND_SUBSURFACE_H
