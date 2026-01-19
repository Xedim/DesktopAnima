#include "App.h"

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/extensions/shape.h>

#include "../scenes/NoiseScene.h"
#include "../scenes/FlowScene.h"

static void makeWindowDesktop(SDL_Window* window) {
    SDL_SysWMinfo wmInfo;
    SDL_VERSION(&wmInfo.version);
    SDL_GetWindowWMInfo(window, &wmInfo);

    Display* display = wmInfo.info.x11.display;
    Window x11Window = wmInfo.info.x11.window;

    // Тип окна = DESKTOP
    Atom wmType = XInternAtom(display, "_NET_WM_WINDOW_TYPE", False);
    Atom wmDesktop = XInternAtom(display, "_NET_WM_WINDOW_TYPE_DESKTOP", False);

    XChangeProperty(
        display,
        x11Window,
        wmType,
        XA_ATOM,
        32,
        PropModeReplace,
        reinterpret_cast<unsigned char*>(&wmDesktop),
        1
    );

    // Опустить ниже всех
    Atom wmState = XInternAtom(display, "_NET_WM_STATE", False);
    Atom wmBelow = XInternAtom(display, "_NET_WM_STATE_BELOW", False);

    XChangeProperty(
        display,
        x11Window,
        wmState,
        XA_ATOM,
        32,
        PropModeAppend,
        reinterpret_cast<unsigned char*>(&wmBelow),
        1
    );

    // Убрать фокус
    XSetInputFocus(display, PointerRoot, RevertToPointerRoot, CurrentTime);

    XFlush(display);
}

void makeClickThrough(SDL_Window* sdlWindow) {
    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    SDL_GetWindowWMInfo(sdlWindow, &info);

    Display* display = info.info.x11.display;
    Window window = info.info.x11.window;

    XRectangle rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = 0;
    rect.height = 0;

    XShapeCombineRectangles(
        display,
        window,
        ShapeInput,
        0, 0,
        &rect,
        1,
        ShapeSet,
        YXBanded
    );

    XFlush(display);
}

App::App() {
    init();
}

App::~App() {
    shutdown();
}

void App::init() {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_CORE
    );

    window = SDL_CreateWindow(
        "Desktop Scenes",
        0, 0,
        1920, 1080,
        SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS
    );

    makeWindowDesktop(window);
    makeClickThrough(window);

    // SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP);

    glContext = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, glContext);

    glewExperimental = GL_TRUE;
    glewInit();

    glDisable(GL_DEPTH_TEST);
    glClearColor(0, 0, 0, 1);

    scenes.set(std::make_unique<NoiseScene>());
}

void App::run() {
    Uint32 last = SDL_GetTicks();

    while (running) {
        Uint32 now = SDL_GetTicks();
        float dt = static_cast<float>(now - last) / 1000.0f;
        last = now;

        handleEvents();
        update(dt);
        render();

        SDL_GL_SwapWindow(window);
    }
}

void App::handleEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT)
            running = false;

        if (e.type == SDL_KEYDOWN) {
            if (e.key.keysym.sym == SDLK_ESCAPE)
                running = false;
            else if (e.key.keysym.sym == SDLK_1)
                scenes.set(std::make_unique<NoiseScene>());
            else if (e.key.keysym.sym == SDLK_2)
                scenes.set(std::make_unique<FlowScene>());
            else
                scenes.onKey(e.key.keysym.sym);
        }
    }
}

void App::update(float dt) const {
    scenes.update(dt);
}

void App::render() const {
    glClear(GL_COLOR_BUFFER_BIT);
    scenes.render();
}

void App::shutdown() {
    scenes.clear();
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

