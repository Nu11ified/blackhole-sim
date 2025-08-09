cmake_minimum_required(VERSION 3.20)
project(blackhole-sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prefer compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_SANITIZERS "Build with address/undefined sanitizers" ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

if(BUILD_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU"))
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# Find raylib via Homebrew (default on macOS)
find_package(raylib REQUIRED)

# Eigen3 header-only
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# ImGui + raylib bindings: we will vendor a minimal integration
add_subdirectory(external/imgui-raylib)

add_library(sim_core INTERFACE)
target_link_libraries(sim_core INTERFACE Eigen3::Eigen)
target_include_directories(sim_core INTERFACE src)

add_executable(blackhole_sim
  src/main.cpp
)
target_link_libraries(blackhole_sim PRIVATE raylib imgui_raylib sim_core)

# macOS frameworks for raylib when Homebrew is used
if(APPLE)
  target_link_libraries(blackhole_sim PRIVATE
    "-framework IOKit" "-framework Cocoa" "-framework OpenGL" "-framework CoreVideo")
endif()

install(TARGETS blackhole_sim RUNTIME DESTINATION bin)


