cmake_minimum_required(VERSION 3.20)
project(DesktopScenes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(Boost REQUIRED)

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
        math/dynamics/Maps.h
        math/dynamics/Maps.cpp
        math/analytics/Analytics.h
        math/analytics/Analytics.cpp
        math/statistics/Statistics.h
        math/statistics/Statistics.cpp
        math/signal/Embedding.h
        math/signal/Embedding.cpp
        math/common/Types.h
        math/common/Constants.h
        math/fractal/Fractal.cpp
        math/fractal/Fractal.h
        math/special/Special.h
        math/special/special.cpp
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
)

