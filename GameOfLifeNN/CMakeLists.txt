project(golnn LANGUAGES CXX)

# --- Simulation core (headless: no GTK/GLEW/OpenGL needed) ---
set(coreSources
	src/Agent.cpp
	src/Simulation.cpp
	src/SpatialGrid.cpp
	src/ThreadPool.cpp
	src/lsRNN/lsRNN.cpp
	src/lsRNN/lsVecTokenizer.cpp
	src/ls3DRNN/ls3DRNN.cpp
)

# Headless simulation library: linked by the GUI, bench, and tests.
add_library(golnn_core STATIC ${coreSources})
target_include_directories(golnn_core PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/src/lsRNN/
	${CMAKE_CURRENT_SOURCE_DIR}/src/ls3DRNN/
	${CMAKE_CURRENT_SOURCE_DIR}/src/
)
target_link_libraries(golnn_core PUBLIC
	glm::glm
	golnn-warnings
)

# --- GUI executable (needs the graphics stack) ---
pkg_check_modules(GTKMM gtkmm-4.0)
pkg_check_modules(SIGC sigc++-3.0)
find_package(GLEW QUIET)
find_package(OpenGL QUIET)
find_package(OpenAL QUIET) # unused today, kept for future audio

if(GTKMM_FOUND AND SIGC_FOUND AND GLEW_FOUND AND OpenGL_FOUND)
	add_executable(golnn
		src/main.cpp
		src/ls3DGraph/ls3DGraph.cpp
	)
	# Project uses no GLU; stop glew.h from requiring the GLU headers
	target_compile_definitions(golnn PRIVATE GLEW_NO_GLU)
	target_include_directories(golnn PRIVATE
		src/ls3DGraph/
		${GTKMM_INCLUDE_DIRS}
		${SIGC_INCLUDE_DIRS}
		${GLEW_INCLUDE_DIRS}
		${OpenAL_INCLUDE_DIRS}
	)
	target_link_directories(golnn PRIVATE ${GTKMM_LIBRARY_DIRS})
	target_link_libraries(golnn PRIVATE
		golnn_core
		${GTKMM_LIBRARIES}
		${SIGC_LIBRARIES}
		GLEW::glew
		OpenGL::GL
	)
else()
	message(STATUS "GUI deps not found (gtkmm-4.0/sigc++-3.0/GLEW/OpenGL) - skipping 'golnn' executable")
	message(STATUS "Headless targets golnn_core, golnn_bench, golnn_smoke are still available")
endif()
