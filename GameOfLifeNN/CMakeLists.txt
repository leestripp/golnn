project(golnn LANGUAGES CXX)

include_directories(
	src/ls3DGraph/
	src/ls3DRNN/
	src/
)

file(GLOB mySources
	src/ls3DGraph/*.cpp
	src/ls3DRNN/*.cpp
	src/*.cpp
)

add_executable(${PROJECT_NAME}
	${mySources}
)

target_link_libraries(${PROJECT_NAME}
	${GTKMM_LIBRARIES}
	${SIGC_LIBRARIES}
	GLEW::glew
	OpenGL::GL
)

