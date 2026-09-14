cmake_minimum_required(VERSION 3.24)

# NVCC 13.2 does not support GCC 16's libstdc++ headers — pin host compiler to gcc-15.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-15)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-15)
endif()

project(water_sim2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.5
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw)

add_executable(water_sim2
    src/main.cpp
    src/kernel.cu
    src/scenes.cpp
    src/scene_desc.cpp
    src/renderer.cpp
    src/shader_program.cpp
    src/particle_pipeline.cpp
    src/screenspace_pipeline.cpp
)
target_link_libraries(water_sim2 PRIVATE
    CUDA::cudart
    OpenGL::GL
    GLEW::GLEW
    glfw
    glm::glm
    imgui_lib
    nlohmann_json::nlohmann_json
)
target_include_directories(water_sim2 PRIVATE src)

# Copy scenes to build directory so the binary can find them
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/scenes)
file(GLOB _scene_jsons CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/scenes/*.json)
foreach(_f ${_scene_jsons})
    configure_file(${_f} ${CMAKE_BINARY_DIR}/scenes/ COPYONLY)
endforeach()

# Copy shaders to build directory as a fallback when SHADERS_DIR is not set
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)
file(GLOB _shader_files CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/shaders/*.vert
    ${CMAKE_SOURCE_DIR}/shaders/*.frag)
foreach(_f ${_shader_files})
    configure_file(${_f} ${CMAKE_BINARY_DIR}/shaders/ COPYONLY)
endforeach()
