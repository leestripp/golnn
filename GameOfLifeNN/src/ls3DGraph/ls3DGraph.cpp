#define GLM_ENABLE_EXPERIMENTAL
#include "ls3DGraph.h"
#include <iostream>
#include <glm/gtx/string_cast.hpp>

// Shaders for the main data, with vertex color attribute
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 view;
uniform mat4 projection;

out vec3 vertexColor;

void main()
{
	gl_Position = projection * view * vec4(aPos, 1.0);
	vertexColor = aColor;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 vertexColor; // Receive color from vertex shader

void main()
{
	FragColor = vec4(vertexColor, 1.0);
}
)";

// Shaders for the grid, using the old uniform color method
const char* gridVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* gridFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 inputColor;

void main()
{
	FragColor = inputColor;
}
)";


ls3DGraph::ls3DGraph() :
	m_camera_pos(0.0f, 2.5f, 75.0f),
	m_camera_front(0.0f, 0.0f, -1.0f),
	m_camera_up(0.0f, 1.0f, 0.0f),
	m_camera_yaw(-90.0f),
	m_camera_pitch(0.0f),
	m_camera_speed(15.0f),
	m_agents_vertex_count(0),
	m_agents_data_dirty(false),
	m_food_vertex_count(0),
	m_food_data_dirty(false),
	m_grid_vertex_count(0),
	m_grid_data_dirty(false),
	m_lmb_down(false)
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

void ls3DGraph::set_agents_data(const vector<vector<double>>& data)
{
	lock_guard<mutex> lock(m_data_mutex);
	m_agents_data = data;
	m_agents_data_dirty = true;
	queue_render();
}

void ls3DGraph::set_food_data(const vector<vector<double>>& data)
{
    lock_guard<mutex> lock(m_data_mutex);
    m_food_data = data;
    m_food_data_dirty = true;
    queue_render();
}

void ls3DGraph::set_grid_data(const vector<vector<double>>& grid_data)
{
	m_grid_data = grid_data;
	m_grid_data_dirty = true;
	queue_render();
}

void ls3DGraph::update_scene(bool key_w, bool key_s, bool key_a, bool key_d, bool is_sprinting)
{
	double delta_time = m_delta_timer.elapsed();
	m_delta_timer.reset();
	
	const float speed_multiplier = is_sprinting ? 4.0f : 1.0f;
	float velocity = m_camera_speed * speed_multiplier * static_cast<float>(delta_time);

	if (key_w) m_camera_pos += m_camera_front * velocity;
	if(key_s) m_camera_pos -= m_camera_front * velocity;
	if(key_a) m_camera_pos -= glm::normalize(glm::cross(m_camera_front, m_camera_up)) * velocity;
	if(key_d) m_camera_pos += glm::normalize(glm::cross(m_camera_front, m_camera_up)) * velocity;
	
	queue_render();
}

void ls3DGraph::on_mouse_motion(double x, double y)
{
	if(m_lmb_down)
	{
		float xoffset = x - m_last_mouse_pos.x;
		float yoffset = m_last_mouse_pos.y - y;
		m_last_mouse_pos = glm::vec2(x, y);
		
		float sensitivity = 0.1f;
		xoffset *= sensitivity;
		yoffset *= sensitivity;
		
		m_camera_yaw   += xoffset;
		m_camera_pitch += yoffset;
		
		if(m_camera_pitch > 89.0f) m_camera_pitch = 89.0f;
		if(m_camera_pitch < -89.0f) m_camera_pitch = -89.0f;
		
		glm::vec3 front;
		front.x = cos(glm::radians(m_camera_yaw)) * cos(glm::radians(m_camera_pitch));
		front.y = sin(glm::radians(m_camera_pitch));
		front.z = sin(glm::radians(m_camera_yaw)) * cos(glm::radians(m_camera_pitch));
		m_camera_front = glm::normalize(front);
	} else
	{
		m_last_mouse_pos = glm::vec2(x, y);
	}
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
	// GTK4 requests a core-profile GL context; GLEW only handles core profiles
	// when experimental mode is enabled first.
	if(const GError* area_err = gtk_gl_area_get_error(GTK_GL_AREA(gobj())))
	{
		cerr << "GLArea context error: " << area_err->message << endl; return;
	}
	if(const GLubyte* ver = glGetString(GL_VERSION))
	{
		cerr << "GL_VERSION: " << reinterpret_cast<const char*>(ver)
		     << " | renderer: " << reinterpret_cast<const char*>(glGetString(GL_RENDERER)) << endl;
	}
	else
	{
		cerr << "GL_VERSION: NULL (no current context after make_current)" << endl;
	}
	glewExperimental = GL_TRUE;
	const GLenum glew_err = glewInit();
	cerr << "glewInit returned code " << static_cast<unsigned>(glew_err)
	     << " (" << reinterpret_cast<const char*>(glewGetErrorString(glew_err)) << ")" << endl;
	// On Wayland, GTK4 hands us an EGL context while GLEW's GLX probe fails
	// with GLEW_ERROR_NO_GLX_DISPLAY (4). GLEW loads all *core* GL function
	// pointers before that probe, so the error is harmless if the pointers
	// we actually use are non-null. Treat it as fatal only when a core
	// entry point is missing.
	if(glew_err != GLEW_OK && glew_err != GLEW_ERROR_NO_GLX_DISPLAY)
	{
		cerr << "Failed to initialize GLEW: " << glewGetErrorString(glew_err) << endl; return;
	}
	// Debug: prove the core entry points survived the failed GLX probe.
	cerr << "GL entry points:"
	     << " glCreateShader=" << (glCreateShader ? "ok" : "NULL")
	     << " glGenVertexArrays=" << (glGenVertexArrays ? "ok" : "NULL")
	     << " glBufferData=" << (glBufferData ? "ok" : "NULL")
	     << " glUseProgram=" << (glUseProgram ? "ok" : "NULL") << endl;
	if(!glCreateShader || !glGenVertexArrays || !glBufferData || !glUseProgram)
	{
		cerr << "Core GL function pointers missing - cannot render." << endl; return;
	}
	glEnable(GL_DEPTH_TEST);
	glPointSize(5.0f);
	
	// Shader setup for MAIN DATA
	GLint success; GLchar infoLog[512];
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vertexShaderSource, NULL); glCompileShader(vs);
	glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
	if(!success)
	{
		glGetShaderInfoLog(vs, 512, NULL, infoLog); cerr << "VS COMPILE ERROR: " << infoLog << endl;
	}
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fragmentShaderSource, NULL); glCompileShader(fs);
	glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
	if(! success)
	{
		glGetShaderInfoLog(fs, 512, NULL, infoLog); cerr << "FS COMPILE ERROR: " << infoLog << endl;
	}
	m_program_id = glCreateProgram();
	glAttachShader(m_program_id, vs); glAttachShader(m_program_id, fs); glLinkProgram(m_program_id);
	glGetProgramiv(m_program_id, GL_LINK_STATUS, &success);
	if(! success)
	{
		glGetProgramInfoLog(m_program_id, 512, NULL, infoLog); cerr << "PROGRAM LINK ERROR: " << infoLog << endl;
	}
	glDeleteShader(vs); glDeleteShader(fs);
	
	// Shader setup for GRID
	GLuint grid_vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(grid_vs, 1, &gridVertexShaderSource, NULL); glCompileShader(grid_vs);
	glGetShaderiv(grid_vs, GL_COMPILE_STATUS, &success);
	if(! success)
	{
		glGetShaderInfoLog(grid_vs, 512, NULL, infoLog); cerr << "GRID VS COMPILE ERROR: " << infoLog << endl;
	}
	GLuint grid_fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(grid_fs, 1, &gridFragmentShaderSource, NULL); glCompileShader(grid_fs);
	glGetShaderiv(grid_fs, GL_COMPILE_STATUS, &success);
	if(! success)
	{
		glGetShaderInfoLog(grid_fs, 512, NULL, infoLog); cerr << "GRID FS COMPILE ERROR: " << infoLog << endl;
	}
	m_grid_program_id = glCreateProgram();
	glAttachShader(m_grid_program_id, grid_vs); glAttachShader(m_grid_program_id, grid_fs); glLinkProgram(m_grid_program_id);
	glGetProgramiv(m_grid_program_id, GL_LINK_STATUS, &success);
	if(! success)
	{
		glGetProgramInfoLog(m_grid_program_id, 512, NULL, infoLog); cerr << "GRID PROGRAM LINK ERROR: " << infoLog << endl;
	}
	glDeleteShader(grid_vs); glDeleteShader(grid_fs);

	// VBO/VAO setup for AGENTS
	glGenVertexArrays(1, &m_agents_vao);
	glGenBuffers(1, &m_agents_vbo);
	glBindVertexArray(m_agents_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_agents_vbo);
	const GLsizei agents_stride = 6 * sizeof(float);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, agents_stride, (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, agents_stride, (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

    // VBO/VAO setup for FOOD
    glGenVertexArrays(1, &m_food_vao);
    glGenBuffers(1, &m_food_vbo);
    glBindVertexArray(m_food_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_food_vbo);
    const GLsizei food_stride = 6 * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, food_stride, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, food_stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
	
	// VBO/VAO setup for GRID (unchanged layout)
	glGenVertexArrays(1, &m_grid_vao);
	glGenBuffers(1, &m_grid_vbo);
	glBindVertexArray(m_grid_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_grid_vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	
	m_delta_timer.start();
}

void ls3DGraph::on_unrealize()
{
	make_current();
	glDeleteVertexArrays(1, &m_agents_vao);
	glDeleteBuffers(1, &m_agents_vbo);
    glDeleteVertexArrays(1, &m_food_vao);
    glDeleteBuffers(1, &m_food_vbo);
	glDeleteVertexArrays(1, &m_grid_vao);
	glDeleteBuffers(1, &m_grid_vbo);
	glDeleteProgram(m_program_id);
	glDeleteProgram(m_grid_program_id);
	Gtk::GLArea::on_unrealize();
}

bool ls3DGraph::on_render(const Glib::RefPtr<Gdk::GLContext>&)
{
	glClearColor(0.0f, 0.1f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	m_projection = glm::perspective(glm::radians(45.0f), (float)get_width() / (float)get_height(), 0.01f, 1000.0f);
	m_view = glm::lookAt(m_camera_pos, m_camera_pos + m_camera_front, m_camera_up);
	
	// Draw the Grid
	glUseProgram(m_grid_program_id);
    glm::mat4 grid_model = glm::mat4(1.0f); 
	glUniformMatrix4fv(glGetUniformLocation(m_grid_program_id, "model"), 1, GL_FALSE, glm::value_ptr(grid_model));
	glUniformMatrix4fv(glGetUniformLocation(m_grid_program_id, "view"), 1, GL_FALSE, glm::value_ptr(m_view));
	glUniformMatrix4fv(glGetUniformLocation(m_grid_program_id, "projection"), 1, GL_FALSE, glm::value_ptr(m_projection));
	
	if(m_grid_data_dirty)
	{
		vector<float> vertices;
		vertices.reserve(m_grid_data.size() * 3);
		for(const auto& row : m_grid_data)
		{
			if(row.size() == 3)
			{
				vertices.push_back(static_cast<float>(row[0]));
				vertices.push_back(static_cast<float>(row[1]));
				vertices.push_back(static_cast<float>(row[2]));
			}
		}
		m_grid_vertex_count = vertices.size() / 3;
		glBindBuffer(GL_ARRAY_BUFFER, m_grid_vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		m_grid_data_dirty = false;
	}
	
	if(m_grid_vertex_count > 0)
	{
		GLint color_loc = glGetUniformLocation(m_grid_program_id, "inputColor");
		glUniform4f(color_loc, 0.3f, 0.3f, 0.3f, 1.0f);
		glBindVertexArray(m_grid_vao);
		glDrawArrays(GL_LINES, 0, m_grid_vertex_count);
	}
	
	// Draw the Agents
	glUseProgram(m_program_id);
	glUniformMatrix4fv(glGetUniformLocation(m_program_id, "view"), 1, GL_FALSE, glm::value_ptr(m_view));
	glUniformMatrix4fv(glGetUniformLocation(m_program_id, "projection"), 1, GL_FALSE, glm::value_ptr(m_projection));
	
	if(m_agents_data_dirty)
	{
		lock_guard<mutex> lock(m_data_mutex);
		vector<float> vertices;
		vertices.reserve(m_agents_data.size() * 6); 
		for(const auto& row : m_agents_data)
		{
			if(row.size() >= 3)
			{
				vertices.push_back(static_cast<float>(row[0]));
				vertices.push_back(static_cast<float>(row[1]));
				vertices.push_back(static_cast<float>(row[2]));
				
				if(row.size() >= 6)
				{
					vertices.push_back(static_cast<float>(row[3]));
					vertices.push_back(static_cast<float>(row[4]));
					vertices.push_back(static_cast<float>(row[5]));
				} else
				{
					vertices.push_back(1.0f);
					vertices.push_back(0.5f);
					vertices.push_back(0.2f);
				}
			}
		}
		m_agents_vertex_count = vertices.size() / 6; 
		glBindBuffer(GL_ARRAY_BUFFER, m_agents_vbo);
		// FIX: Use GL_DYNAMIC_DRAW for data that changes frequently
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
		m_agents_data_dirty = false;
	}
	
	if(m_agents_vertex_count > 0)
	{
		glBindVertexArray(m_agents_vao);
        // The draw style for agents is hardcoded to polygons (triangles)
        glDrawArrays(GL_TRIANGLES, 0, m_agents_vertex_count);
	}

    // Draw the Food
    // Food data needs to be loaded into its own buffer and VAO
    if(m_food_data_dirty)
	{
		lock_guard<mutex> lock(m_data_mutex);
		vector<float> vertices;
		vertices.reserve(m_food_data.size() * 6);
		for(const auto& row : m_food_data)
		{
			if(row.size() >= 3)
			{
				vertices.push_back(static_cast<float>(row[0]));
				vertices.push_back(static_cast<float>(row[1]));
				vertices.push_back(static_cast<float>(row[2]));
				
				if(row.size() >= 6)
				{
					vertices.push_back(static_cast<float>(row[3]));
					vertices.push_back(static_cast<float>(row[4]));
					vertices.push_back(static_cast<float>(row[5]));
				} else
				{
					vertices.push_back(0.2f);
					vertices.push_back(0.8f);
					vertices.push_back(0.2f);
				}
			}
		}
		m_food_vertex_count = vertices.size() / 6;
		glBindBuffer(GL_ARRAY_BUFFER, m_food_vbo);
		// FIX: Use GL_DYNAMIC_DRAW for data that changes frequently
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
		m_food_data_dirty = false;
	}

    if(m_food_vertex_count > 0)
    {
        glBindVertexArray(m_food_vao);
        // Food is drawn as points
        glDrawArrays(GL_POINTS, 0, m_food_vertex_count);
    }
	
	glBindVertexArray(0);
	return true;
}
