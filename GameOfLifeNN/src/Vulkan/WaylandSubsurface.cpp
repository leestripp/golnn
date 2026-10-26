#include "WaylandSubsurface.h"

#include <wayland-client.h>
#include "viewporter-client-protocol.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>

struct WaylandSubsurface::Impl
{
	wl_display* display = nullptr;
	wl_surface* child = nullptr;
	wl_surface* parent = nullptr;
	wl_subsurface* subsurface = nullptr;
	wp_viewport* viewport = nullptr;

	~Impl() { shutdown(); }

	static wl_subcompositor* bind_subcompositor(wl_display& native_display)
	{
		wl_subcompositor* result = nullptr;
		wl_registry* registry = wl_display_get_registry(&native_display);
		static const wl_registry_listener listener{
			[](void* data, wl_registry* callback_registry, std::uint32_t name,
				const char* interface, const std::uint32_t version) {
				if(std::string_view(interface) == "wl_subcompositor")
					*static_cast<wl_subcompositor**>(data) = static_cast<wl_subcompositor*>(
						wl_registry_bind(callback_registry, name, &wl_subcompositor_interface, std::min(1U, version)));
			},
			[](void*, wl_registry*, std::uint32_t) {}};
		wl_registry_add_listener(registry, &listener, &result);
		wl_display_roundtrip(&native_display);
		wl_registry_destroy(registry);
		return result;
	}

	static wp_viewporter* bind_viewporter(wl_display& native_display)
	{
		wp_viewporter* result = nullptr;
		wl_registry* registry = wl_display_get_registry(&native_display);
		static const wl_registry_listener listener{
			[](void* data, wl_registry* callback_registry, std::uint32_t name,
				const char* interface, const std::uint32_t version) {
				if(std::string_view(interface) == "wp_viewporter")
					*static_cast<wp_viewporter**>(data) = static_cast<wp_viewporter*>(
						wl_registry_bind(callback_registry, name, &wp_viewporter_interface, std::min(1U, version)));
			},
			[](void*, wl_registry*, std::uint32_t) {}};
		wl_registry_add_listener(registry, &listener, &result);
		wl_display_roundtrip(&native_display);
		wl_registry_destroy(registry);
		return result;
	}

	void initialize(wl_display& native_display, wl_compositor& compositor, wl_surface& parent_surface)
	{
		display = &native_display;
		parent = &parent_surface;
		wl_subcompositor* subcompositor = bind_subcompositor(native_display);
		wp_viewporter* viewporter = bind_viewporter(native_display);
		if(subcompositor == nullptr || viewporter == nullptr)
			throw std::runtime_error("Wayland subcompositor/viewporter is unavailable");
		child = wl_compositor_create_surface(&compositor);
		if(child == nullptr)
		{
			wl_subcompositor_destroy(subcompositor);
			throw std::runtime_error("Could not create Vulkan child surface");
		}
		subsurface = wl_subcompositor_get_subsurface(subcompositor, child, parent);
		wl_subcompositor_destroy(subcompositor);
		if(subsurface == nullptr) throw std::runtime_error("Could not create Vulkan subsurface");
		viewport = wp_viewporter_get_viewport(viewporter, child);
		wp_viewporter_destroy(viewporter);
		if(viewport == nullptr) throw std::runtime_error("Could not create Vulkan logical viewport");
		wl_subsurface_set_desync(subsurface);
		wl_subsurface_set_position(subsurface, 0, 0);
		// Let GTK's placeholder receive keyboard/mouse input rather than the
		// native child surface.
		wl_region* empty_region = wl_compositor_create_region(&compositor);
		wl_surface_set_input_region(child, empty_region);
		wl_region_destroy(empty_region);
		wl_surface_commit(parent);
		wl_display_flush(display);
	}

	void shutdown()
	{
		if(viewport != nullptr) { wp_viewport_destroy(viewport); viewport = nullptr; }
		if(subsurface != nullptr) { wl_subsurface_destroy(subsurface); subsurface = nullptr; }
		if(child != nullptr) { wl_surface_destroy(child); child = nullptr; }
		if(display != nullptr) { wl_display_flush(display); display = nullptr; }
		parent = nullptr;
	}
};

WaylandSubsurface::WaylandSubsurface() : m_impl(std::make_unique<Impl>()) {}
WaylandSubsurface::~WaylandSubsurface() = default;
bool WaylandSubsurface::initialize(wl_display& display, wl_compositor& compositor,
	wl_surface& parent, std::string& error)
{
	try { shutdown(); m_impl->initialize(display, compositor, parent); return true; }
	catch(const std::exception& exception) { error = exception.what(); shutdown(); return false; }
}
void WaylandSubsurface::set_position(const int x, const int y)
{
	if(m_impl->subsurface != nullptr)
	{
		wl_subsurface_set_position(m_impl->subsurface, x, y);
		wl_surface_commit(m_impl->parent);
		wl_display_flush(m_impl->display);
	}
}
void WaylandSubsurface::set_logical_size(const int width, const int height)
{
	if(m_impl->viewport != nullptr)
	{
		wp_viewport_set_destination(m_impl->viewport, width, height);
		wl_surface_commit(m_impl->child);
		wl_display_flush(m_impl->display);
	}
}
wl_surface& WaylandSubsurface::surface() const { return *m_impl->child; }
void WaylandSubsurface::shutdown() { m_impl->shutdown(); }
