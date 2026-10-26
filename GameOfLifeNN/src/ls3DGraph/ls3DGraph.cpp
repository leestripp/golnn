#define GLM_ENABLE_EXPERIMENTAL
#include "ls3DGraph.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>

namespace
{
	constexpr std::size_t AGENT_VERTEX_COUNT = 9;
	constexpr float AGENT_MESH[] = {
		 0.0f,  0.5f,  0.0f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,
		 0.0f,  0.5f,  0.0f,  0.5f, -0.5f,  0.5f,  0.0f, -0.5f, -0.5f,
		-0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.0f, -0.5f, -0.5f
	};

	const char* agent_vertex_shader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in mat4 aTransform;
layout(location = 6) in mat4 aPreviousTransform;
uniform mat4 view;
uniform mat4 projection;
uniform float interpolation_alpha;
out vec3 vertexColor;
void main()
{
	mat4 transform = aPreviousTransform * (1.0 - interpolation_alpha)
		+ aTransform * interpolation_alpha;
	gl_Position = projection * view * transform * vec4(aPos, 1.0);
	vertexColor = aColor.rgb;
}
)";

	const char* color_vertex_shader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 view;
uniform mat4 projection;
out vec3 vertexColor;
void main()
{
	gl_Position = projection * view * vec4(aPos, 1.0);
	vertexColor = aColor;
}
)";

	const char* color_fragment_shader = R"(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;
void main() { FragColor = vec4(vertexColor, 1.0); }
)";

	const char* grid_vertex_shader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 view;
uniform mat4 projection;
void main() { gl_Position = projection * view * vec4(aPos, 1.0); }
)";

	const char* grid_fragment_shader = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 inputColor;
void main() { FragColor = inputColor; }
)";

	GLuint compile_shader(const GLenum type, const char* source)
	{
		const GLuint shader = glCreateShader(type);
		glShaderSource(shader, 1, &source, nullptr);
		glCompileShader(shader);
		GLint success = GL_FALSE;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if(success == GL_FALSE)
		{
			char log[512];
			glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
			std::cerr << "Shader compile error: " << log << '\n';
			glDeleteShader(shader);
			return 0;
		}
		return shader;
	}

	GLuint make_program(const char* vertex_source, const char* fragment_source)
	{
		const GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
		const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
		if(vertex == 0 || fragment == 0) return 0;
		const GLuint program = glCreateProgram();
		glAttachShader(program, vertex);
		glAttachShader(program, fragment);
		glLinkProgram(program);
		glDeleteShader(vertex);
		glDeleteShader(fragment);
		GLint success = GL_FALSE;
		glGetProgramiv(program, GL_LINK_STATUS, &success);
		if(success == GL_FALSE)
		{
			char log[512];
			glGetProgramInfoLog(program, sizeof(log), nullptr, log);
			std::cerr << "Program link error: " << log << '\n';
			glDeleteProgram(program);
			return 0;
		}
		return program;
	}

	void set_matrix_attribute(const GLuint location, const std::size_t offset)
	{
		constexpr std::size_t column_size = 4 * sizeof(float);
		for(GLuint column = 0; column < 4; ++column)
		{
			glVertexAttribPointer(location + column, 4, GL_FLOAT, GL_FALSE,
				sizeof(AgentRenderInstance), reinterpret_cast<void*>(offset + column * column_size));
			glEnableVertexAttribArray(location + column);
			glVertexAttribDivisor(location + column, 1);
		}
	}
}

static_assert(sizeof(AgentRenderInstance) == 36 * sizeof(float));
static_assert(sizeof(FoodRenderInstance) == 6 * sizeof(float));

ls3DGraph::ls3DGraph()
	: m_camera_pos(0.0f, 2.5f, 75.0f),
	m_camera_front(0.0f, 0.0f, -1.0f),
	m_camera_up(0.0f, 1.0f, 0.0f),
	m_camera_yaw(-90.0f),
	m_camera_pitch(0.0f),
	m_camera_speed(15.0f)
{
	set_required_version(4, 0);
	set_use_es(false);
	set_has_depth_buffer(true);
	set_auto_render(false);

	signal_realize().connect(sigc::mem_fun(*this, &ls3DGraph::on_realize));
	signal_unrealize().connect(sigc::mem_fun(*this, &ls3DGraph::on_unrealize));
	signal_render().connect(sigc::mem_fun(*this, &ls3DGraph::on_render), false);

	m_motion_controller = Gtk::EventControllerMotion::create();
	m_motion_controller->signal_motion().connect(sigc::mem_fun(*this, &ls3DGraph::on_mouse_motion), false);
	add_controller(m_motion_controller);
	m_button_controller = Gtk::EventControllerLegacy::create();
	m_button_controller->signal_event().connect(sigc::mem_fun(*this, &ls3DGraph::on_mouse_button), false);
	add_controller(m_button_controller);
}

void ls3DGraph::set_render_snapshot(std::shared_ptr<const WorldSnapshot> snapshot)
{
	if(snapshot == m_snapshot)
	{
		queue_render();
		return;
	}
	m_snapshot = std::move(snapshot);
	m_scene_data_dirty = true;
	queue_render();
}

void ls3DGraph::set_grid_data(const std::vector<glm::vec3>& grid_data)
{
	m_grid_data = grid_data;
	m_grid_data_dirty = true;
	queue_render();
}

void ls3DGraph::update_scene(const bool key_w, const bool key_s, const bool key_a,
                             const bool key_d, const bool is_sprinting)
{
	const double delta_time = m_delta_timer.elapsed();
	m_delta_timer.reset();
	const float speed_multiplier = is_sprinting ? 4.0f : 1.0f;
	const float velocity = m_camera_speed * speed_multiplier * static_cast<float>(delta_time);
	if(key_w) m_camera_pos += m_camera_front * velocity;
	if(key_s) m_camera_pos -= m_camera_front * velocity;
	if(key_a) m_camera_pos -= glm::normalize(glm::cross(m_camera_front, m_camera_up)) * velocity;
	if(key_d) m_camera_pos += glm::normalize(glm::cross(m_camera_front, m_camera_up)) * velocity;
	queue_render();
}

void ls3DGraph::on_mouse_motion(const double x, const double y)
{
	if(m_lmb_down)
	{
		float xoffset = static_cast<float>(x) - m_last_mouse_pos.x;
		float yoffset = m_last_mouse_pos.y - static_cast<float>(y);
		m_last_mouse_pos = glm::vec2(x, y);
		constexpr float sensitivity = 0.1f;
		xoffset *= sensitivity;
		yoffset *= sensitivity;
		m_camera_yaw += xoffset;
		m_camera_pitch = std::clamp(m_camera_pitch + yoffset, -89.0f, 89.0f);
		glm::vec3 front;
		front.x = cos(glm::radians(m_camera_yaw)) * cos(glm::radians(m_camera_pitch));
		front.y = sin(glm::radians(m_camera_pitch));
		front.z = sin(glm::radians(m_camera_yaw)) * cos(glm::radians(m_camera_pitch));
		m_camera_front = glm::normalize(front);
	}
	else m_last_mouse_pos = glm::vec2(x, y);
}

bool ls3DGraph::on_mouse_button(const Glib::RefPtr<const Gdk::Event>& event)
{
	if(event->get_event_type() == Gdk::Event::Type::BUTTON_PRESS && event->get_button() == 1)
	{
		m_lmb_down = true;
		return true;
	}
	if(event->get_event_type() == Gdk::Event::Type::BUTTON_RELEASE && event->get_button() == 1)
	{
		m_lmb_down = false;
		return true;
	}
	return false;
}

void ls3DGraph::on_realize()
{
	Gtk::GLArea::on_realize();
	make_current();
	if(const GError* area_err = gtk_gl_area_get_error(GTK_GL_AREA(gobj())))
	{
		std::cerr << "GLArea context error: " << area_err->message << '\n';
		return;
	}
	if(const GLubyte* version = glGetString(GL_VERSION))
	{
		std::cerr << "GL_VERSION: " << reinterpret_cast<const char*>(version)
		          << " | renderer: " << reinterpret_cast<const char*>(glGetString(GL_RENDERER)) << '\n';
	}
	glewExperimental = GL_TRUE;
	const GLenum glew_err = glewInit();
	std::cerr << "glewInit returned code " << static_cast<unsigned>(glew_err)
	          << " (" << reinterpret_cast<const char*>(glewGetErrorString(glew_err)) << ")\n";
	if(glew_err != GLEW_OK && glew_err != GLEW_ERROR_NO_GLX_DISPLAY) return;
	if(!glCreateShader || !glGenVertexArrays || !glBufferData || !glUseProgram ||
	   !glDrawArraysInstanced || !glVertexAttribDivisor)
	{
		std::cerr << "Required instanced OpenGL entry points are missing.\n";
		return;
	}
	std::cerr << "Instanced OpenGL entry points: OK\n";

	glEnable(GL_DEPTH_TEST);
	glPointSize(5.0f);
	m_agent_program_id = make_program(agent_vertex_shader, color_fragment_shader);
	m_food_program_id = make_program(color_vertex_shader, color_fragment_shader);
	m_grid_program_id = make_program(grid_vertex_shader, grid_fragment_shader);
	if(!m_agent_program_id || !m_food_program_id || !m_grid_program_id) return;

	glGenVertexArrays(1, &m_agents_vao);
	glBindVertexArray(m_agents_vao);
	glGenBuffers(1, &m_agent_mesh_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_agent_mesh_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(AGENT_MESH), AGENT_MESH, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);
	glGenBuffers(1, &m_agent_instance_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_agent_instance_vbo);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(AgentRenderInstance),
		reinterpret_cast<void*>(offsetof(AgentRenderInstance, color)));
	glEnableVertexAttribArray(1);
	glVertexAttribDivisor(1, 1);
	set_matrix_attribute(2, offsetof(AgentRenderInstance, transform));
	set_matrix_attribute(6, offsetof(AgentRenderInstance, previous_transform));

	glGenVertexArrays(1, &m_food_vao);
	glBindVertexArray(m_food_vao);
	glGenBuffers(1, &m_food_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_food_vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FoodRenderInstance),
		reinterpret_cast<void*>(offsetof(FoodRenderInstance, position)));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(FoodRenderInstance),
		reinterpret_cast<void*>(offsetof(FoodRenderInstance, color)));
	glEnableVertexAttribArray(1);

	glGenVertexArrays(1, &m_grid_vao);
	glBindVertexArray(m_grid_vao);
	glGenBuffers(1, &m_grid_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_grid_vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	m_delta_timer.start();
}

void ls3DGraph::on_unrealize()
{
	// The widget is already becoming unrealized; its GL context is still valid.
	if(const auto context = get_context()) context->make_current();
	glDeleteVertexArrays(1, &m_agents_vao);
	glDeleteBuffers(1, &m_agent_mesh_vbo);
	glDeleteBuffers(1, &m_agent_instance_vbo);
	glDeleteVertexArrays(1, &m_food_vao);
	glDeleteBuffers(1, &m_food_vbo);
	glDeleteVertexArrays(1, &m_grid_vao);
	glDeleteBuffers(1, &m_grid_vbo);
	glDeleteProgram(m_agent_program_id);
	glDeleteProgram(m_food_program_id);
	glDeleteProgram(m_grid_program_id);
	Gtk::GLArea::on_unrealize();
}

bool ls3DGraph::on_render(const Glib::RefPtr<Gdk::GLContext>&)
{
	glClearColor(0.0f, 0.1f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m_projection = glm::perspective(glm::radians(45.0f),
		static_cast<float>(get_width()) / static_cast<float>(get_height()), 0.01f, 1000.0f);
	m_view = glm::lookAt(m_camera_pos, m_camera_pos + m_camera_front, m_camera_up);

	glUseProgram(m_grid_program_id);
	glUniformMatrix4fv(glGetUniformLocation(m_grid_program_id, "view"), 1, GL_FALSE, glm::value_ptr(m_view));
	glUniformMatrix4fv(glGetUniformLocation(m_grid_program_id, "projection"), 1, GL_FALSE, glm::value_ptr(m_projection));
	glUniform4f(glGetUniformLocation(m_grid_program_id, "inputColor"), 0.3f, 0.3f, 0.3f, 1.0f);
	if(m_grid_data_dirty)
	{
		m_grid_vertex_count = m_grid_data.size();
		glBindBuffer(GL_ARRAY_BUFFER, m_grid_vbo);
		glBufferData(GL_ARRAY_BUFFER, m_grid_data.size() * sizeof(glm::vec3), m_grid_data.data(), GL_STATIC_DRAW);
		m_grid_data_dirty = false;
	}
	if(m_grid_vertex_count > 0)
	{
		glBindVertexArray(m_grid_vao);
		glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_grid_vertex_count));
	}

	if(m_scene_data_dirty && m_snapshot)
	{
		m_agent_instance_count = m_snapshot->agents.size();
		m_food_vertex_count = m_snapshot->food.size();
		if(m_agent_instance_count > 0)
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_agent_instance_vbo);
			glBufferData(GL_ARRAY_BUFFER, m_agent_instance_count * sizeof(AgentRenderInstance),
				m_snapshot->agents.data(), GL_DYNAMIC_DRAW);
		}
		if(m_food_vertex_count > 0)
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_food_vbo);
			glBufferData(GL_ARRAY_BUFFER, m_food_vertex_count * sizeof(FoodRenderInstance),
				m_snapshot->food.data(), GL_DYNAMIC_DRAW);
		}
		m_scene_data_dirty = false;
	}

	if(m_agent_instance_count > 0)
	{
		glUseProgram(m_agent_program_id);
		glUniformMatrix4fv(glGetUniformLocation(m_agent_program_id, "view"), 1, GL_FALSE, glm::value_ptr(m_view));
		glUniformMatrix4fv(glGetUniformLocation(m_agent_program_id, "projection"), 1, GL_FALSE, glm::value_ptr(m_projection));
		const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
		const float alpha = m_snapshot
			? std::clamp(static_cast<float>((now - m_snapshot->publication_time_seconds) / 0.033), 0.0f, 1.0f) : 1.0f;
		glUniform1f(glGetUniformLocation(m_agent_program_id, "interpolation_alpha"), alpha);
		glBindVertexArray(m_agents_vao);
		glDrawArraysInstanced(GL_TRIANGLES, 0, AGENT_VERTEX_COUNT, static_cast<GLsizei>(m_agent_instance_count));
	}

	if(m_food_vertex_count > 0)
	{
		glUseProgram(m_food_program_id);
		glUniformMatrix4fv(glGetUniformLocation(m_food_program_id, "view"), 1, GL_FALSE, glm::value_ptr(m_view));
		glUniformMatrix4fv(glGetUniformLocation(m_food_program_id, "projection"), 1, GL_FALSE, glm::value_ptr(m_projection));
		glBindVertexArray(m_food_vao);
		glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_food_vertex_count));
	}
	glBindVertexArray(0);
	return true;
}
