#pragma once
#include <SDL2/SDL.h>
#include "../app/SceneManager.h"

class App {
public:
    App();
    ~App();

    void run();

private:
    void init();
    void handleEvents();
    void update(float dt) const;
    void render() const;
    void shutdown();

    bool running = true;
    SDL_Window* window = nullptr;
    SDL_GLContext glContext = nullptr;

    SceneManager scenes;
};
