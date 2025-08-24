cmake_minimum_required(VERSION 3.28)

project(myProject LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cppcheck
find_program(CPPCHECK_EXECUTABLE cppcheck REQUIRED)
add_custom_target(cppcheck
    COMMAND ${CPPCHECK_EXECUTABLE}
            --enable=warning
            ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} # Analyze source and potentially generated files
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Cppcheck..."
    VERBATIM
)
set_property(TARGET cppcheck PROPERTY ENVIRONMENT "CPPCHECK_EXITCODE=2")

# pkg-config
include(FindPkgConfig)

# sigc++
pkg_check_modules(SIGC REQUIRED sigc++-3.0)
include_directories(${SIGC_INCLUDE_DIRS})

# openAL
find_package(OpenAL REQUIRED)
include_directories(${OpenAL_INCLUDE_DIRS})

# Find the GLEW library
find_package(GLEW REQUIRED)

# Find the GLM header-only library
# This will find the include directory for glm
find_package(glm REQUIRED)

# Find the core OpenGL library
find_package(OpenGL REQUIRED)

# gtkmm
pkg_check_modules(GTKMM REQUIRED gtkmm-4.0)
include_directories(${GTKMM_INCLUDE_DIRS})
link_directories(${GTKMM_LIBRARY_DIRS})

include_directories(
	${GLEW_INCLUDE_DIRS}
	${glm_INCLUDE_DIR}
)

add_subdirectory(GameOfLifeNN)
