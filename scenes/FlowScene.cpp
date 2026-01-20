#include "FlowScene.h"
#include "../gl/Shader.h"
#include "../gl/Quad.h"
#include <GL/glew.h>
#include <random>

void FlowScene::onEnter() {
    shaders[FlowVariant::Flow] = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "flow/flow.frag"
    );
    shaders[FlowVariant::Colorful] = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "flow/flow_colorful.frag"
    );
    shaders[FlowVariant::ColorChange] = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "flow/flow_color_change.frag"
    );
    shaders[FlowVariant::Space] = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "flow/flow_space.frag"
    );

    blitShader = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "flow/blit.frag"
    );
    quad = std::make_unique<Quad>();
    initTrail(1920, 1080);

    std::uniform_real_distribution<float> seedDist(0.0f, 1000.0f);
    params.seed = seedDist(rng);
    params.time = 0.0f;

    currentVariant = FlowVariant::ColorChange;
}

void FlowScene::initTrail(int width, int height) {
    glGenFramebuffers(2, trailFBO);
    glGenTextures(2, trailTex);

    for (int i = 0; i < 2; ++i) {
        glBindTexture(GL_TEXTURE_2D, trailTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, trailTex[i], 0);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FlowScene::update(float dt) {
    params.time += dt;

    params.color.r = up.r_level + up.r_amplitude * sinf(params.time * up.r_frequency + up.r_phase);
    params.color.g = up.g_level + up.g_amplitude * sinf(params.time * up.g_frequency + up.g_phase);
    params.color.b = up.b_level + up.b_amplitude * sinf(params.time * up.b_frequency + up.b_phase);

    params.trailDecay = up.td_level + up.td_amplitude * cosf(params.time * up.td_frequency + up.td_phase);
    params.speed      = up.speed_level + up.speed_amplitude * cosf(params.time * up.speed_frequency + up.speed_phase);
    params.intensity  = up.intensity_level + up.intensity_amplitude * cosf(params.time * up.intensity_frequency + up.intensity_phase);
}

void FlowScene::render() {
    int src = ping;
    int dst = 1 - ping;

    glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[dst]);
    glViewport(0, 0, 1920, 1080);

    shaders[currentVariant]->use();
    shaders[currentVariant]->setFloat("u_seed", params.seed);
    shaders[currentVariant]->setFloat("u_time", params.time);
    shaders[currentVariant]->setVec2("u_resolution", 1920.0f, 1080.0f);
    shaders[currentVariant]->setFloat("u_scale", params.scale);
    shaders[currentVariant]->setFloat("u_speed", params.speed);
    shaders[currentVariant]->setVec3("u_color", params.color.r, params.color.g, params.color.b);
    shaders[currentVariant]->setFloat("u_trailDecay", params.trailDecay);
    shaders[currentVariant]->setFloat("u_intensity", params.intensity);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, trailTex[src]);
    shaders[currentVariant]->setInt("u_trailTex", 0);

    quad->draw();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 1920, 1080);

    blitShader->use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, trailTex[dst]);
    blitShader->setInt("u_tex", 0);

    quad->draw();

    ping = dst;
}

void FlowScene::nextVariant() {
    int v = static_cast<int>(currentVariant);
    v = (v + 1) % shaders.size();
    currentVariant = static_cast<FlowVariant>(v);

    for (int i = 0; i < 2; ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[i]);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    params.time = 0.0f;
}

void FlowScene::prevVariant() {
    int v = static_cast<int>(currentVariant);
    v = (v - 1 + shaders.size()) % shaders.size();
    currentVariant = static_cast<FlowVariant>(v);

    for (int i = 0; i < 2; ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[i]);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    params.time = 0.0f;
    params.seed = 0.0f;
    ping = 0;

    params.time = 0.0f;
    std::uniform_real_distribution<float> seedDist(0.0f, 1000.0f);
    params.seed = seedDist(rng);
}

void FlowScene::onKey(SDL_Keycode key) {
    if (key == SDLK_q) prevVariant();
    if (key == SDLK_e) nextVariant();
}
