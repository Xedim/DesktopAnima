cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0167 NEW)
cmake_policy(SET CMP0135 NEW)
project(DesktopScenes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(Boost REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS math_tr1 math_c99)

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
)

add_library(MathCore
        math/common/Types.h
        math/common/Constants.h
        math/common/Utils.h

        math/pattern/MathEngine.h
        math/pattern/PatternIdentity.h
        math/pattern/PatternExecutor.h
        math/pattern/PatternList.h
        math/pattern/PatternInit.h
        math/pattern/Functions.h
        math/pattern/Functions.cpp
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


include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(testME
        tests/testME.cpp
)

target_link_libraries(testME PRIVATE
        MathCore
        gtest
        gtest_main
)

enable_testing()
add_test(NAME MathEngineTests COMMAND testME)
