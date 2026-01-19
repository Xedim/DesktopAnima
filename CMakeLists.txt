cmake_minimum_required(VERSION 3.20)
project(DesktopScenes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)

add_definitions(-DSHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/")

add_executable(DesktopScenes
        main.cpp
        app/App.cpp
        app/SceneManager.cpp
        scenes/NoiseScene.cpp
        scenes/FlowScene.cpp
        scenes/GrayScottScene.cpp
        gl/Shader.cpp
        gl/Quad.cpp
        scenes/FlowScene.h
)

target_include_directories(DesktopScenes PRIVATE
        .
        /usr/include
)

target_link_libraries(DesktopScenes PRIVATE
        SDL2::SDL2
        GLEW::GLEW
        OpenGL::GL
        X11::X11
        X11::Xext
)

