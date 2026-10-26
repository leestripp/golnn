#ifndef LS3DGRAPH_H
#define LS3DGRAPH_H

#include <gtkmm/glarea.h>
#include <gtkmm/eventcontrollermotion.h>
#include <gtkmm/eventcontrollerlegacy.h>
#include <gdkmm/event.h>
#include <glibmm/timer.h>

#include <memory>
#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "RenderSnapshot.h"

class ls3DGraph : public Gtk::GLArea
{
public:
	ls3DGraph();

	void set_render_snapshot(std::shared_ptr<const WorldSnapshot> snapshot);
	void set_grid_data(const std::vector<glm::vec3>& grid_data);
	void update_scene(bool key_w, bool key_s, bool key_a, bool key_d, bool is_sprinting);

private:
	void on_realize();
	void on_unrealize();
	bool on_render(const Glib::RefPtr<Gdk::GLContext>& context);
	void on_mouse_motion(double x, double y);
	bool on_mouse_button(const Glib::RefPtr<const Gdk::Event>& event);

	GLuint m_agent_program_id = 0;
	GLuint m_food_program_id = 0;
	GLuint m_grid_program_id = 0;

	GLuint m_agents_vao = 0;
	GLuint m_agent_mesh_vbo = 0;
	GLuint m_agent_instance_vbo = 0;
	GLuint m_food_vao = 0;
	GLuint m_food_vbo = 0;
	GLuint m_grid_vao = 0;
	GLuint m_grid_vbo = 0;

	glm::mat4 m_view;
	glm::mat4 m_projection;
	glm::vec3 m_camera_pos;
	glm::vec3 m_camera_front;
	glm::vec3 m_camera_up;
	float m_camera_yaw;
	float m_camera_pitch;
	float m_camera_speed;

	std::shared_ptr<const WorldSnapshot> m_snapshot;
	bool m_scene_data_dirty = true;
	std::size_t m_agent_instance_count = 0;
	std::size_t m_food_vertex_count = 0;
	std::size_t m_grid_vertex_count = 0;
	bool m_grid_data_dirty = false;
	std::vector<glm::vec3> m_grid_data;

	Glib::RefPtr<Gtk::EventControllerMotion> m_motion_controller;
	Glib::RefPtr<Gtk::EventControllerLegacy> m_button_controller;
	bool m_lmb_down = false;
	glm::vec2 m_last_mouse_pos;
	Glib::Timer m_delta_timer;
};

#endif // LS3DGRAPH_H
