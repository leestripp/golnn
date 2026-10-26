#include "ls3DGraph.h"
#include "Simulation.h"
#include "Agent.h"

// gtkmm 4
#include <gtkmm/application.h>
#include <gtkmm/window.h>
#include <gtkmm/eventcontrollerkey.h>
#include <glibmm/timer.h>
#include <glibmm/main.h>
#include <gtkmm/box.h>
#include <gtkmm/label.h>

#include <sigc++/connection.h>
#include <gdk/gdkkeysyms.h>
#include <map>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>

// A simple 3D model for our agent (e.g., a tetrahedron)
const std::vector<glm::vec3> agent_model = {
    { 0.0f,  0.5f,  0.0f}, // Top
    {-0.5f, -0.5f,  0.5f}, // Front-left
    { 0.5f, -0.5f,  0.5f}, // Front-right
    
    { 0.0f,  0.5f,  0.0f}, // Top
    { 0.5f, -0.5f,  0.5f}, // Front-right
    { 0.0f, -0.5f, -0.5f}, // Back
    
    { -0.5f, -0.5f,  0.5f}, // Front-left
    { 0.5f, -0.5f,  0.5f}, // Front-right
    { 0.0f, -0.5f, -0.5f}  // Back
};

// Encapsulates the entire application state and logic
class Game
{
public:
    Game();
    int run();
	
private:
    // This handler will be called when the application starts up.
    void on_startup();
    // Callback for the timeout
    bool on_timer_tick();
    // Handlers for key press and release events using EventControllerKey
    bool on_key_pressed(guint keyval, guint, Gdk::ModifierType state);
    void on_key_released(guint keyval, guint, Gdk::ModifierType state);
    // Helper function to set up the world grid
    void setup_grid();

    Glib::RefPtr<Gtk::Application> m_app;
    Gtk::Window m_window;
	
    // UI Widgets
    Gtk::Box m_main_box;
    Gtk::Label m_info_label;
    Gtk::Label m_stats_label;
    ls3DGraph m_graph;
	
    Simulation m_simulation;
    // Holds the connection to the timer so it can be disconnected later if needed.
    sigc::connection m_timer_connection;
	
    // Boolean flags to track key presses
    bool m_key_w = false, m_key_a = false, m_key_s = false, m_key_d = false, m_key_shift = false;
	
    // Simulation timing variables
    static constexpr int TICK_RATE_MS = 33;
    Glib::Timer m_tick_timer;
    double m_tick_accumulator = 0.0;
};

Game::Game() :
	m_app(Gtk::Application::create("com.lscppdev.golnn", Gio::Application::Flags::DEFAULT_FLAGS)),
	m_simulation(100, 50, 100.0f)
{
	// We now connect a signal handler for when the application starts up.
	// The main window and all its children will be created here.
    m_app->signal_startup().connect(
        sigc::mem_fun(*this, &Game::on_startup)
    );
}

void Game::on_startup()
{
    // The window and timer connections must be made after the application has started up.
    m_window.set_default_size(1280, 720);
	
    // Layout
    // 1. Configure the main vertical box container.
    m_main_box.set_orientation(Gtk::Orientation::VERTICAL);
    m_main_box.set_spacing(5);
	
    // 2. Configure the info label with key bindings.
    m_info_label.set_text("Controls: [W][A][S][D] to Move Camera | [LMB+Drag] to Look | [Shift] to Sprint | [E] to Spawn Agents | [R] to Spawn Food | [Esc] to Quit");
    m_info_label.set_wrap(true);
    m_info_label.set_selectable(true);
	
    // 3. IMPORTANT: Configure the GLArea to expand and fill available space.
    m_graph.set_expand(true);
	
    // 4. Add the GLArea and the Label to the box container.
    m_main_box.append(m_graph);
    m_main_box.append(m_info_label);
    m_main_box.append(m_stats_label);
	
    // 5. Set the box as the main window's child, replacing the old direct connection.
    m_window.set_child(m_main_box);
	
	// Add window to the app
    m_app->add_window(m_window);
	
    // This is required to receive keyboard events
    m_window.set_can_focus();
    m_window.grab_focus();
    
    // Use the modern Gtk::EventControllerKey to handle keyboard input.
    auto controller = Gtk::EventControllerKey::create();
    controller->signal_key_pressed().connect(
        sigc::mem_fun(*this, &Game::on_key_pressed), false
    );
    controller->signal_key_released().connect(
        sigc::mem_fun(*this, &Game::on_key_released), false
    );
    m_window.add_controller(controller);
	
    setup_grid();
	
    // The timer is now connected to the window's 'realize' signal.
    // This ensures the timer only starts once the window is fully set up.
    m_window.signal_realize().connect(
        [this]() {
            // Connect the timer and store the connection object.
            m_tick_timer.start();
            m_timer_connection = Glib::signal_timeout().connect(
                sigc::mem_fun(*this, &Game::on_timer_tick),
                1); // Use a low timeout for rendering
        }
    );
    
    // Disconnect timer on window unrealize
    m_window.signal_unrealize().connect(
        [this]() {
            m_timer_connection.disconnect();
        }
    );
	
    // We need to show the window after all setup is done.
    m_window.show();
}

int Game::run()
{
    return m_app->run();
}

void Game::setup_grid()
{
    const float world_size = m_simulation.m_world_size;
    std::vector<std::vector<double>> grid_data;
    float half_size = world_size / 2.0f;
    for(int i = 0; i <= world_size; i += 5)
	{
        float z_pos = -half_size + i;
        // Z-axis lines
        grid_data.push_back({-half_size, 0.0, z_pos});
        grid_data.push_back({half_size, 0.0, z_pos});
        // X-axis lines
        grid_data.push_back({z_pos, 0.0, -half_size});
        grid_data.push_back({z_pos, 0.0, half_size});
    }
    m_graph.set_grid_data(grid_data);
}

bool Game::on_key_pressed(guint keyval, guint, Gdk::ModifierType)
{
    // Check for both lowercase and uppercase variants of the keys.
    switch (keyval)
	{
        case GDK_KEY_w: case GDK_KEY_W: m_key_w = true; break;
        case GDK_KEY_s: case GDK_KEY_S: m_key_s = true; break;
        case GDK_KEY_a: case GDK_KEY_A: m_key_a = true; break;
        case GDK_KEY_d: case GDK_KEY_D: m_key_d = true; break;
        case GDK_KEY_Shift_L:
        case GDK_KEY_Shift_R:
            m_key_shift = true;
            break;
        case GDK_KEY_1: m_simulation.set_brain_tier(BrainTier::SMALL); break;
        case GDK_KEY_2: m_simulation.set_brain_tier(BrainTier::MEDIUM); break;
        case GDK_KEY_3: m_simulation.set_brain_tier(BrainTier::LARGE); break;
        // Press 'e' to spawn 20 new agents with the selected brain tier
        case GDK_KEY_e: case GDK_KEY_E:
            m_simulation.spawn_new_agents(20);
			break;
		// spawn food 'r'
        case GDK_KEY_r: case GDK_KEY_R:
            m_simulation.spawn_new_food(100, 0.1f);
            break;
        case GDK_KEY_Escape:
            m_window.close();
            break;
    }
    return true; // We handled the event.
}

void Game::on_key_released(guint keyval, guint, Gdk::ModifierType)
{
    // Check for both lowercase and uppercase variants of the keys.
    switch (keyval)
	{
        case GDK_KEY_w: case GDK_KEY_W: m_key_w = false; break;
        case GDK_KEY_s: case GDK_KEY_S: m_key_s = false; break;
        case GDK_KEY_a: case GDK_KEY_A: m_key_a = false; break;
        case GDK_KEY_d: case GDK_KEY_D: m_key_d = false; break;
        case GDK_KEY_Shift_L:
        case GDK_KEY_Shift_R:
            m_key_shift = false;
            break;
    }
}

bool Game::on_timer_tick()
{
    // Update camera every frame, as this should be as smooth as possible
    m_graph.update_scene(m_key_w, m_key_s, m_key_a, m_key_d, m_key_shift);
	
    // Fixed-step simulation update
    m_tick_accumulator += m_tick_timer.elapsed() * 1000.0;
    m_tick_timer.reset();
	
    while(m_tick_accumulator >= TICK_RATE_MS)
	{
        m_simulation.tick();
		
		// debug
        // std::cout << "Timer tick. Agents: " << m_simulation.m_agents.size() << ", Food: " << m_simulation.m_food.size() << std::endl;
        
        m_tick_accumulator -= TICK_RATE_MS;
    }
	
    std::vector<std::vector<double>> agent_render_data;
    std::vector<std::vector<double>> food_render_data;
    std::lock_guard<std::mutex> lock(m_simulation.m_data_mutex);
	
    // Get agent data
    for(const auto& agent : m_simulation.m_agents)
	{
        for (const auto& model_vertex : agent_model)
		{
            glm::vec4 world_vertex = agent.m_transform * glm::vec4(model_vertex, 1.0f);
            // Use the agent's state color for rendering
            agent_render_data.push_back({
                world_vertex.x, world_vertex.y, world_vertex.z,
                static_cast<double>(agent.m_color.r),
                static_cast<double>(agent.m_color.g),
                static_cast<double>(agent.m_color.b)
            });
        }
    }
	
    // Get food data (render as points with type-specific colors)
    for(const auto& food : m_simulation.m_food)
	{
		double r = 0.2, g = 0.8, b = 0.2; // Default to green (VEGETABLE)
		if (food.m_food_type == FoodType::MEAT) {
			r = 0.8; g = 0.2; b = 0.2; // Red for MEAT
		}
        food_render_data.push_back({
            (double)food.position.x, (double)food.position.y, (double)food.position.z,
            r, g, b
        });
    }
	
    const EcologyStats stats = m_simulation.ecology_stats();
    m_stats_label.set_text(
        "Tick " + std::to_string(stats.tick) +
        " | Feeders " + std::to_string(stats.feeders) +
        " | Predators " + std::to_string(stats.predators) +
        " | Vegetation " + std::to_string(stats.vegetables) +
        " | Meat " + std::to_string(stats.meat) +
        " | Births " + std::to_string(stats.total_births) +
        " | Deaths " + std::to_string(stats.total_deaths) +
        " | Workers " + std::to_string(m_simulation.worker_count()) +
        " | Next brain " + std::string(brain_tier_name(m_simulation.get_config().brain_tier)) +
        " [S " + std::to_string(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::SMALL)]) +
        " M " + std::to_string(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::MEDIUM)]) +
        " L " + std::to_string(stats.brain_tier_counts[static_cast<std::size_t>(BrainTier::LARGE)]) + "]");

    // Now we can use the new ls3DGraph methods to render both data sets.
    m_graph.set_agents_data(agent_render_data);
    m_graph.set_food_data(food_render_data);
    
    return true;
}


int main()
{
    Game game;
    return game.run();
}

