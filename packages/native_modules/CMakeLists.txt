cmake_minimum_required(VERSION 3.26)

project("monodepth_native_modules" LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

set(PYBIND11_FINDPYTHON ON)

add_subdirectory(common)
add_subdirectory(align_depth_ransac)
