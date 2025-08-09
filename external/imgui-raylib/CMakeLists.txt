cmake_minimum_required(VERSION 3.20)
project(imgui-raylib LANGUAGES CXX)

add_library(imgui_raylib
  imgui_impl_raylib.h
  imgui_impl_raylib.cpp
)

target_include_directories(imgui_raylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(imgui_raylib PUBLIC raylib)


