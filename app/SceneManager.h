#pragma once
#include <memory>
#include "Scene.h"

class SceneManager {
public:
    void set(std::unique_ptr<Scene> scene);
    void update(float dt) const;
    void render() const;
    void clear();

private:
    std::unique_ptr<Scene> current;
};
