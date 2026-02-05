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
        math/analytics/Analytics.h
        math/analytics/Analytics.cpp
)

add_executable(testME
        tests/testME.cpp
        tests/Functions/testAnalytics.cpp
        tests/Functions/testAlgebra.cpp
        tests/Functions/testExpLog.cpp
        tests/Functions/testTrigonometric.cpp
        tests/Functions/textHyperbolic.cpp
        tests/Functions/testHybrid.cpp
        tests/Functions/testSpecial.cpp
        tests/Functions/testGeneralNumerical.cpp
        tests/Functions/testFractals.cpp
        tests/Functions/testIteration.cpp
        tests/Functions/testStatDesc.cpp
        tests/Functions/testStatShape.cpp
        tests/Functions/testStatQuantile.cpp
        tests/Functions/testStatRobust.cpp
        tests/Functions/testStatCorrelations.cpp
        tests/Functions/testStatDistributions.cpp
        tests/Functions/testStatInfo.cpp
        tests/Functions/testStatCharacter.cpp
        tests/Functions/testStatTimeSeries.cpp
        tests/Functions/testStatSampling.cpp
        tests/Functions/testStatRegression.cpp
        tests/Functions/testOutliers.cpp
        tests/Functions/testStatTests.cpp
        tests/Functions/uniTest.cpp
        tests/testMathEngine.cpp
        tests/testTypes.cpp
)

target_compile_definitions(DesktopScenes PRIVATE
        SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/"
)

target_include_directories(DesktopScenes PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(DesktopScenes PRIVATE
        MathCore
        SDL2::SDL2
        GLEW::GLEW
        OpenGL::GL
        X11::X11
        X11::Xext
)


find_package(Boost CONFIG REQUIRED COMPONENTS math_tr1 math_c99)


target_link_libraries(MathCore PUBLIC
        Boost::boost
        Boost::math_tr1
        Boost::math_c99
)

target_include_directories(MathCore PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/math
)


include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

target_link_libraries(testME PRIVATE
        MathCore
        gtest
        gtest_main
)
