cmake_minimum_required(VERSION 3.20)
project(DesktopScenes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS math_tr1 math_c99)

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
        Utils.cpp
        math/pattern/Pattern1D.h
        math/pattern/Pattern2D.h
        math/pattern/Pattern1D.cpp
        math/pattern/Pattern2D.cpp
        math/analytics/Analytics.h
        math/analytics/Analytics.cpp
        math/common/Types.h
        math/common/Constants.h
        math/common/Utils.h
        math/common/todo.h
        math/pattern/PatternRegistry.h
        math/pattern/PatternIdentity.h
        math/pattern/Functions.cpp
        math/pattern/Functions.h
        math/pattern/PatternExecutor.h
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
        Boost::boost
        Boost::math_tr1
        Boost::math_c99
)

