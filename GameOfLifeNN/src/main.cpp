#include "ls3DGraph.h"
#include "Simulation.h"

#ifdef GOLNN_HAS_VULKAN
#include "Vulkan/VulkanRenderer.h"
#include "Vulkan/WaylandSubsurface.h"
#include <gdk/wayland/gdkwayland.h>
#include <cstdlib>
#include <string_view>
#endif

#include <gtkmm/application.h>
#include <gtkmm/window.h>
#include <gtkmm/eventcontrollerkey.h>
#include <gtkmm/gesturedrag.h>
#include <glibmm/main.h>
#include <gtkmm/box.h>
#include <gtkmm/label.h>
#include <gdkmm/graphene_point.h>

#include <sigc++/connection.h>
#include <gdk/gdkkeysyms.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stop_token>
#include <thread>

namespace
{
	std::size_t initial_agent_count()
	{
		const char* value = std::getenv("GOLNN_INITIAL_AGENTS");
		if(value == nullptr) return 100;
		return std::clamp<std::size_t>(std::strtoul(value, nullptr, 10), 1, 10000);
	}
}

class Game
{
public:
	Game();
	int run();

private:
	void on_startup();
	bool on_timer_tick();
	bool on_key_pressed(guint keyval, guint, Gdk::ModifierType state);
	void on_key_released(guint keyval, guint, Gdk::ModifierType state);
	void setup_world_bounds();
	void run_simulation(std::stop_token stop_token);
	void stop_simulation();
#ifdef GOLNN_HAS_VULKAN
	void update_vulkan();
	void fallback_to_opengl(std::string_view reason);
#endif

	Glib::RefPtr<Gtk::Application> m_app;
	Gtk::Window m_window;
	Gtk::Box m_main_box;
	Gtk::Label m_info_label;
	Gtk::Label m_stats_label;
	ls3DGraph m_graph;
	std::vector<glm::vec3> m_grid_data;
#ifdef GOLNN_HAS_VULKAN
	Gtk::Box m_vulkan_placeholder;
	WaylandSubsurface m_wayland_subsurface;
	VulkanRenderer m_vulkan_renderer;
	bool m_use_vulkan = false;
	bool m_vulkan_ready = false;
	int m_vulkan_x = 0;
	int m_vulkan_y = 0;
	double m_drag_offset_x = 0.0;
	double m_drag_offset_y = 0.0;
	int m_vulkan_width = 0;
	int m_vulkan_height = 0;
#endif
	Simulation m_simulation;
	sigc::connection m_timer_connection;
	std::jthread m_simulation_thread;

	bool m_key_w = false;
	bool m_key_a = false;
	bool m_key_s = false;
	bool m_key_d = false;
	bool m_key_shift = false;
	static constexpr int TICK_RATE_MS = 33;
};

Game::Game()
	: m_app(Gtk::Application::create("com.lscppdev.golnn", Gio::Application::Flags::DEFAULT_FLAGS)),
	m_simulation(initial_agent_count(), 50, 100.0f)
{
	m_app->signal_startup().connect(sigc::mem_fun(*this, &Game::on_startup));
#ifdef GOLNN_HAS_VULKAN
	const char* requested = std::getenv("GOLNN_RENDERER");
	m_use_vulkan = requested == nullptr
		|| (std::string_view(requested) != "opengl" && std::string_view(requested) != "gl");
#endif
}

void Game::on_startup()
{
	m_window.set_default_size(1280, 720);
	m_main_box.set_orientation(Gtk::Orientation::VERTICAL);
	m_main_box.set_spacing(5);
	m_info_label.set_text("Controls: [W][A][S][D] Camera | [LMB+Drag] Look | [Shift] Sprint | "
		"[E] Spawn Agents | [R] Spawn Food | [1/2/3] Brain Tier | [Esc] Quit");
	m_info_label.set_wrap(true);
	m_info_label.set_selectable(true);
#ifdef GOLNN_HAS_VULKAN
	if(m_use_vulkan)
	{
		m_window.set_title("GameOfLifeNN - Vulkan");
		m_vulkan_placeholder.set_expand(true);
		m_main_box.append(m_vulkan_placeholder);
	}
	else
#endif
	{
		m_window.set_title("GameOfLifeNN - OpenGL");
		m_graph.set_expand(true);
		m_main_box.append(m_graph);
	}
	m_main_box.append(m_info_label);
	m_main_box.append(m_stats_label);
	m_window.set_child(m_main_box);
	m_app->add_window(m_window);
	m_window.set_can_focus();
	m_window.grab_focus();

	auto controller = Gtk::EventControllerKey::create();
	controller->signal_key_pressed().connect(sigc::mem_fun(*this, &Game::on_key_pressed), false);
	controller->signal_key_released().connect(sigc::mem_fun(*this, &Game::on_key_released), false);
	m_window.add_controller(controller);
#ifdef GOLNN_HAS_VULKAN
	auto drag = Gtk::GestureDrag::create();
	drag->set_button(1);
	drag->set_exclusive(true);
	drag->signal_drag_begin().connect([this](double, double) {
		m_drag_offset_x = 0.0;
		m_drag_offset_y = 0.0;
	});
	drag->signal_drag_update().connect([this](const double x, const double y) {
		if(!m_use_vulkan) return;
		m_vulkan_renderer.mouse_drag_delta(x - m_drag_offset_x, y - m_drag_offset_y);
		m_drag_offset_x = x;
		m_drag_offset_y = y;
	});
	drag->signal_drag_end().connect([this](double, double) {
		m_drag_offset_x = 0.0;
		m_drag_offset_y = 0.0;
	});
	m_vulkan_placeholder.add_controller(drag);
#endif
	setup_world_bounds();

	m_window.signal_realize().connect([this]() {
		m_timer_connection = Glib::signal_timeout().connect(
			sigc::mem_fun(*this, &Game::on_timer_tick), 4);
		m_simulation_thread = std::jthread([this](const std::stop_token stop_token) {
			run_simulation(stop_token);
		});
	});
	m_window.signal_unrealize().connect([this]() {
		m_timer_connection.disconnect();
#ifdef GOLNN_HAS_VULKAN
		m_vulkan_renderer.shutdown();
		m_wayland_subsurface.shutdown();
#endif
		stop_simulation();
	});
	m_window.show();
#ifdef GOLNN_HAS_VULKAN
	// Optional smoke-test hooks: normal interactive runs leave both unset.
	if(const char* value = std::getenv("GOLNN_TEST_RESIZE_MS"))
	{
		const int delay = std::atoi(value);
		if(delay > 0) Glib::signal_timeout().connect([this]() {
			m_window.set_default_size(900, 600);
			return false;
		}, static_cast<unsigned int>(delay));
	}
	if(const char* value = std::getenv("GOLNN_AUTO_CLOSE_MS"))
	{
		const int delay = std::atoi(value);
		if(delay > 0) Glib::signal_timeout().connect([this]() {
			m_window.close();
			return false;
		}, static_cast<unsigned int>(delay));
	}
#endif
}

int Game::run()
{
	const int result = m_app->run();
	stop_simulation();
	return result;
}

void Game::stop_simulation()
{
	if(m_simulation_thread.joinable())
	{
		m_simulation_thread.request_stop();
		m_simulation_thread.join();
	}
}

void Game::run_simulation(const std::stop_token stop_token)
{
	using Clock = std::chrono::steady_clock;
	constexpr auto tick_duration = std::chrono::milliseconds(TICK_RATE_MS);
	auto next_tick = Clock::now();
	try
	{
		while(!stop_token.stop_requested())
		{
			m_simulation.tick();
			next_tick += tick_duration;
			const auto now = Clock::now();
			if(now - next_tick > tick_duration * 4) next_tick = now + tick_duration;
			std::this_thread::sleep_until(next_tick);
		}
	}
	catch(const std::exception& error)
	{
		std::cerr << "Simulation thread stopped after an error: " << error.what() << '\n';
	}
	catch(...)
	{
		std::cerr << "Simulation thread stopped after an unknown error.\n";
	}
}

void Game::setup_world_bounds()
{
	const float half = m_simulation.m_world_size / 2.0f;
	m_grid_data.clear();
	m_grid_data.reserve(24); // 12 cube edges, two vertices each
	// Four edges parallel to each axis. No central floor grid or wall mesh:
	// the agents' universe is outlined without obscuring the scene.
	for(const float y : {-half, half}) for(const float z : {-half, half})
	{
		m_grid_data.push_back({-half, y, z});
		m_grid_data.push_back({half, y, z});
	}
	for(const float x : {-half, half}) for(const float z : {-half, half})
	{
		m_grid_data.push_back({x, -half, z});
		m_grid_data.push_back({x, half, z});
	}
	for(const float x : {-half, half}) for(const float y : {-half, half})
	{
		m_grid_data.push_back({x, y, -half});
		m_grid_data.push_back({x, y, half});
	}
	m_graph.set_grid_data(m_grid_data);
#ifdef GOLNN_HAS_VULKAN
	m_vulkan_renderer.set_grid_data(m_grid_data);
#endif
}

#ifdef GOLNN_HAS_VULKAN
void Game::fallback_to_opengl(const std::string_view reason)
{
	if(!reason.empty()) std::cerr << "Vulkan unavailable; using OpenGL: " << reason << '\n';
	m_vulkan_renderer.shutdown();
	m_wayland_subsurface.shutdown();
	m_use_vulkan = false;
	m_vulkan_ready = false;
	m_main_box.remove(m_vulkan_placeholder);
	m_graph.set_expand(true);
	m_main_box.prepend(m_graph);
	m_graph.set_grid_data(m_grid_data);
	m_graph.set_render_snapshot(m_simulation.render_snapshot());
	m_window.set_title("GameOfLifeNN - OpenGL");
}

void Game::update_vulkan()
{
	const auto widget_origin = m_vulkan_placeholder.compute_point(
		m_window, Gdk::Graphene::Point(0.0f, 0.0f));
	if(!widget_origin) return;
	double surface_transform_x = 0.0;
	double surface_transform_y = 0.0;
	m_window.get_surface_transform(surface_transform_x, surface_transform_y);
	const int scene_x = static_cast<int>(std::lround(widget_origin->get_x() + surface_transform_x));
	const int scene_y = static_cast<int>(std::lround(widget_origin->get_y() + surface_transform_y));
	if(!m_vulkan_ready)
	{
		const int width = m_vulkan_placeholder.get_width();
		const int height = m_vulkan_placeholder.get_height();
		if(width <= 1 || height <= 1) return;
		auto gdk_display = Gdk::Display::get_default();
		auto gdk_surface = m_window.get_surface();
		if(!gdk_display || !gdk_surface
			|| !GDK_IS_WAYLAND_DISPLAY(gdk_display->gobj())
			|| !GDK_IS_WAYLAND_SURFACE(gdk_surface->gobj()))
		{
			fallback_to_opengl("a native Wayland surface is required");
			return;
		}
		wl_display* display = gdk_wayland_display_get_wl_display(gdk_display->gobj());
		wl_compositor* compositor = gdk_wayland_display_get_wl_compositor(gdk_display->gobj());
		wl_surface* parent = gdk_wayland_surface_get_wl_surface(gdk_surface->gobj());
		if(display == nullptr || compositor == nullptr || parent == nullptr)
		{
			fallback_to_opengl("native Wayland handles are unavailable");
			return;
		}
		std::string error;
		if(!m_wayland_subsurface.initialize(*display, *compositor, *parent, error)
			|| !m_vulkan_renderer.initialize(*display, m_wayland_subsurface.surface(),
				static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), error))
		{
			fallback_to_opengl(error);
			return;
		}
		m_wayland_subsurface.set_position(scene_x, scene_y);
		m_wayland_subsurface.set_logical_size(width, height);
		m_vulkan_x = scene_x;
		m_vulkan_y = scene_y;
		m_vulkan_width = width;
		m_vulkan_height = height;
		m_vulkan_ready = true;
		std::cout << "Renderer: native Vulkan/Wayland subsurface\n";
		std::cout << "Vulkan scene geometry: position=" << scene_x << ',' << scene_y
			<< " size=" << width << 'x' << height
			<< " surface=" << gdk_surface_get_width(gdk_surface->gobj()) << 'x'
			<< gdk_surface_get_height(gdk_surface->gobj())
			<< " transform=" << surface_transform_x << ',' << surface_transform_y
			<< " scale=" << gdk_surface_get_scale(gdk_surface->gobj()) << '\n';
	}
	if(scene_x != m_vulkan_x || scene_y != m_vulkan_y)
	{
		m_vulkan_x = scene_x;
		m_vulkan_y = scene_y;
		m_wayland_subsurface.set_position(scene_x, scene_y);
	}
	const int width = m_vulkan_placeholder.get_width();
	const int height = m_vulkan_placeholder.get_height();
	if(width > 1 && height > 1 && (width != m_vulkan_width || height != m_vulkan_height))
	{
		m_vulkan_width = width;
		m_vulkan_height = height;
		m_wayland_subsurface.set_logical_size(width, height);
		m_vulkan_renderer.resize(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
	}
	m_vulkan_renderer.update_scene(m_key_w, m_key_s, m_key_a, m_key_d, m_key_shift);
	std::string error;
	if(!m_vulkan_renderer.render(error)) fallback_to_opengl(error);
}


#endif

bool Game::on_key_pressed(const guint keyval, guint, Gdk::ModifierType)
{
	switch(keyval)
	{
		case GDK_KEY_w: case GDK_KEY_W: m_key_w = true; break;
		case GDK_KEY_s: case GDK_KEY_S: m_key_s = true; break;
		case GDK_KEY_a: case GDK_KEY_A: m_key_a = true; break;
		case GDK_KEY_d: case GDK_KEY_D: m_key_d = true; break;
		case GDK_KEY_Shift_L:
		case GDK_KEY_Shift_R: m_key_shift = true; break;
		case GDK_KEY_1: m_simulation.set_brain_tier(BrainTier::SMALL); break;
		case GDK_KEY_2: m_simulation.set_brain_tier(BrainTier::MEDIUM); break;
		case GDK_KEY_3: m_simulation.set_brain_tier(BrainTier::LARGE); break;
		case GDK_KEY_e: case GDK_KEY_E: m_simulation.spawn_new_agents(20); break;
		case GDK_KEY_r: case GDK_KEY_R: m_simulation.spawn_new_food(100, 0.1f); break;
		case GDK_KEY_Escape: m_window.close(); break;
	}
	return true;
}

void Game::on_key_released(const guint keyval, guint, Gdk::ModifierType)
{
	switch(keyval)
	{
		case GDK_KEY_w: case GDK_KEY_W: m_key_w = false; break;
		case GDK_KEY_s: case GDK_KEY_S: m_key_s = false; break;
		case GDK_KEY_a: case GDK_KEY_A: m_key_a = false; break;
		case GDK_KEY_d: case GDK_KEY_D: m_key_d = false; break;
		case GDK_KEY_Shift_L:
		case GDK_KEY_Shift_R: m_key_shift = false; break;
	}
}

bool Game::on_timer_tick()
{
#ifdef GOLNN_HAS_VULKAN
	if(m_use_vulkan) update_vulkan();
	else
#endif
		m_graph.update_scene(m_key_w, m_key_s, m_key_a, m_key_d, m_key_shift);
	const auto snapshot = m_simulation.render_snapshot();
	if(!snapshot) return true;
#ifdef GOLNN_HAS_VULKAN
	if(m_use_vulkan) m_vulkan_renderer.set_render_snapshot(snapshot);
	else
#endif
		m_graph.set_render_snapshot(snapshot);

	const EcologyStats& stats = snapshot->stats;
	m_stats_label.set_text(
		"Tick " + std::to_string(stats.tick) +
		" | Feeders " + std::to_string(stats.feeders) +
		" | Predators " + std::to_string(stats.predators) +
		" | Vegetation " + std::to_string(stats.vegetables) +
		" | Meat " + std::to_string(stats.meat) +
		" | Births " + std::to_string(stats.total_births) +
		" | Deaths " + std::to_string(stats.total_deaths) +
		" | Workers " + std::to_string(snapshot->worker_count) +
		" | Autonomy " + std::to_string(static_cast<int>(stats.mean_autonomy * 100.0)) + "%" +
		" | Rest " + std::to_string(stats.state_counts[4]) +
		" Graze " + std::to_string(stats.state_counts[5]) +
		" Explore " + std::to_string(stats.state_counts[6]) +
		" Investigate " + std::to_string(stats.state_counts[7]) +
		" | Next brain " + std::string(brain_tier_name(snapshot->selected_brain_tier)) +
		" [S " + std::to_string(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::SMALL)]) +
		" M " + std::to_string(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::MEDIUM)]) +
		" L " + std::to_string(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::LARGE)]) + "]"
#ifdef GOLNN_HAS_VULKAN
		+ (m_use_vulkan ? " | Vulkan" : " | OpenGL")
#endif
	);
	return true;
}

int main()
{
	Game game;
	return game.run();
}
