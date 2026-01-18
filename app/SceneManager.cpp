#include "SceneManager.h"

void SceneManager::set(std::unique_ptr<Scene> scene) {
    if (current) {
        current->onExit();
    }

    current = std::move(scene);

    if (current) {
        current->onEnter();
    }
}

void SceneManager::update(float dt) const {
    if (current) current->update(dt);
}

void SceneManager::render() const {
    if (current) current->render();
}

void SceneManager::clear() {
    if (current) {
        current->onExit();
        current.reset();
    }
}
