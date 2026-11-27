cmake_minimum_required(VERSION 3.18)
project(main LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glfw
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glad
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/imgui
)

set(SOURCES
    main.cu
    3rdparty/glad/gl.c
    3rdparty/imgui/imgui.cpp
    3rdparty/imgui/imgui_demo.cpp
    3rdparty/imgui/imgui_draw.cpp
    3rdparty/imgui/imgui_tables.cpp
    3rdparty/imgui/imgui_widgets.cpp
    3rdparty/imgui/imgui_impl_glfw.cpp
    3rdparty/imgui/imgui_impl_opengl3.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Zc:preprocessor>
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glfw/glfw3.lib 
    CUDA::cudart
)

target_link_libraries(${PROJECT_NAME} PRIVATE opengl32 gdi32)