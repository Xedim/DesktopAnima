#pragma once
#include <SDL2/SDL.h>

class Scene {
public:
    virtual ~Scene() = default;

    virtual void onEnter() {}
    virtual void onExit() {}

    virtual void update(float dt) = 0;
    virtual void render() = 0;

    virtual void onKey(SDL_Keycode key) {}
};
