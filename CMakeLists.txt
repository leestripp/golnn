cmake_minimum_required(VERSION 3.28)

project(myProject LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared warning flags for all golnn targets (ratchet: add more over time)
add_library(golnn-warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	target_compile_options(golnn-warnings INTERFACE
		-Wall -Wextra -Wpedantic -Wshadow -Wdeprecated
	)
endif()

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

# glm (header-only): used by the simulation core AND the GUI
find_package(glm REQUIRED)

add_subdirectory(GameOfLifeNN)

# --- Headless benchmark: ticks/sec without GTK/GL ---
add_executable(golnn_bench bench/bench.cpp)
target_link_libraries(golnn_bench PRIVATE golnn_core golnn-warnings)

# --- Tests ---
# Framework-free unit tests (test/test_framework.h); one executable per
# component, each registered with CTest.
enable_testing()

add_executable(golnn_smoke test/smoke_test.cpp)
target_link_libraries(golnn_smoke PRIVATE golnn_core golnn-warnings)
add_test(NAME smoke COMMAND golnn_smoke)

add_executable(golnn_test_spatial_grid test/spatial_grid_test.cpp)
target_link_libraries(golnn_test_spatial_grid PRIVATE golnn_core golnn-warnings)
target_include_directories(golnn_test_spatial_grid PRIVATE test)
add_test(NAME spatial_grid COMMAND golnn_test_spatial_grid)

add_executable(golnn_test_lsrnn test/lsrnn_test.cpp)
target_link_libraries(golnn_test_lsrnn PRIVATE golnn_core golnn-warnings)
target_include_directories(golnn_test_lsrnn PRIVATE test)
add_test(NAME lsrnn COMMAND golnn_test_lsrnn)

add_executable(golnn_test_ls3drnn test/ls3drnn_test.cpp)
target_link_libraries(golnn_test_ls3drnn PRIVATE golnn_core golnn-warnings)
target_include_directories(golnn_test_ls3drnn PRIVATE test)
add_test(NAME ls3drnn COMMAND golnn_test_ls3drnn)

add_executable(golnn_test_lsvec_tokenizer test/lsvec_tokenizer_test.cpp)
target_link_libraries(golnn_test_lsvec_tokenizer PRIVATE golnn_core golnn-warnings)
target_include_directories(golnn_test_lsvec_tokenizer PRIVATE test)
add_test(NAME lsvec_tokenizer COMMAND golnn_test_lsvec_tokenizer)
