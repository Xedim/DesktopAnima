#pragma once
#include "../app/Scene.h"
#include "../gl/Shader.h"
#include "../gl/Quad.h"
#include <memory>

class NoiseScene : public Scene {
public:
    void onEnter() override;
    void update(float dt) override;
    void render() override;

private:
    std::unique_ptr<Shader> shader;
    std::unique_ptr<Quad> quad;
    float time = 0.0f;
    float seed = 0.0f;
};

