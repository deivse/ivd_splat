set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(module_name "align_depth_ransac")
set(native_target "_align_depth_ransac")

file(GLOB_RECURSE RANSAC_SRC CONFIGURE_DEPENDS src/*.cpp)
pybind11_add_module(${native_target} ${RANSAC_SRC})

target_include_directories(${native_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(Eigen3 REQUIRED CONFIG)
target_link_libraries(${native_target} PRIVATE Eigen3::Eigen common)

# Install Python package files
install(DIRECTORY src/${module_name}/ 
        DESTINATION ${module_name}
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi")

# Install the native module
install(TARGETS ${native_target} DESTINATION ${module_name})
