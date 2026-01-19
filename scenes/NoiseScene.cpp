#include "NoiseScene.h"
#include "../gl/Shader.h"
#include "../gl/Quad.h"
#include <random>

void NoiseScene::onEnter() {
    shader = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "noise.frag"
    );
    quad = std::make_unique<Quad>();
    time = 0.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    seed = dist(gen);
}

void NoiseScene::update(float dt) {
    time += dt;
}

void NoiseScene::render() {
    shader->use();
    shader->setFloat("u_seed", seed);
    shader->setFloat("u_time", time);
    shader->setVec2("u_resolution", 1920.0f, 1080.0f);
    shader->setVec3("u_color", 0.5f, 0.0f, 0.5f);
    shader->setFloat("u_scale", 20.0f);
    shader->setInt("u_octaves", 5);
    shader->setFloat("u_speed", 0.35f);
    shader->setFloat("u_ampFactor", 0.50f);

    quad->draw();
}

