#ifndef LS3DGRAPH_H
#define LS3DGRAPH_H

#include <gtkmm/glarea.h>
#include <gtkmm/eventcontrollermotion.h>
#include <gtkmm/eventcontrollerlegacy.h>
#include <gdkmm/event.h>
#include <glibmm/timer.h>

#include <vector>
#include <mutex>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

// This enum defines the primitive types for drawing
enum class DrawStyle {
	POINTS,
	LINES,
	POLYGONS
};

// This struct holds the data for a single object to be rendered
struct Renderable {
    vector<vector<double>> data;
    DrawStyle style;
};

class ls3DGraph : public Gtk::GLArea
{
public:
	ls3DGraph();
	
    // The main data set for the simulation (e.g., agents as polygons)
	void set_agents_data(const vector<vector<double>>& data);
    // A separate data set for food (e.g., points)
    void set_food_data(const vector<vector<double>>& data);
	void set_grid_data(const vector<vector<double>>& grid_data);
    void update_scene(bool key_w, bool key_s, bool key_a, bool key_d, bool is_sprinting);
	
private:
	void on_realize();
	void on_unrealize();
	bool on_render(const Glib::RefPtr<Gdk::GLContext>& context);
	
	void on_mouse_motion(double x, double y);
    bool on_mouse_button(const Glib::RefPtr<const Gdk::Event>& event);
	
	GLuint m_program_id;
    // Shader program specifically for the grid
	GLuint m_grid_program_id; 
	
	// Buffers for agents and food
    GLuint m_agents_vbo, m_agents_vao;
    GLuint m_food_vbo, m_food_vao;

	// Buffers for the grid
	GLuint m_grid_vbo;
	GLuint m_grid_vao;
	
	glm::mat4 m_view;
	glm::mat4 m_projection;
	
    glm::vec3 m_camera_pos;
    glm::vec3 m_camera_front;
    glm::vec3 m_camera_up;
    float m_camera_yaw;
    float m_camera_pitch;
    float m_camera_speed;
	
    // Main data properties
	size_t m_agents_vertex_count;
    bool m_agents_data_dirty;
	vector<vector<double>> m_agents_data;
	mutex m_data_mutex;

    size_t m_food_vertex_count;
    bool m_food_data_dirty;
    vector<vector<double>> m_food_data;
	
    // Grid data properties
	size_t m_grid_vertex_count;
    bool m_grid_data_dirty;
    vector<vector<double>> m_grid_data;
	
    // Input Controllers
    Glib::RefPtr<Gtk::EventControllerMotion> m_motion_controller;
    Glib::RefPtr<Gtk::EventControllerLegacy> m_button_controller;
	
    // Input State
    bool m_lmb_down;
    glm::vec2 m_last_mouse_pos;
	
    // Timer
    Glib::Timer m_delta_timer;
};

#endif // LS3DGRAPH_H
