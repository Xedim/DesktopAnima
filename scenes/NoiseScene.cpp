#include "NoiseScene.h"
#include "../gl/Shader.h"
#include "../gl/Quad.h"
#include <random>

struct params {
    float time;
    float seed;
    struct Color { float r, g, b; } color;
    float scale;
    float speed;
    float intensity;

    // Анимация для цвета и интенсивности
    struct {
        float r_level, r_amplitude, r_frequency, r_phase;
        float g_level, g_amplitude, g_frequency, g_phase;
        float b_level, b_amplitude, b_frequency, b_phase;
        float speed_level, speed_amplitude, speed_frequency, speed_phase;
        float intensity_level, intensity_amplitude, intensity_frequency, intensity_phase;
    } up;
};

void NoiseScene::onEnter() {
    quad = std::make_unique<Quad>();

    for (int i = 0; i < static_cast<int>(NoiseVariant::_Count); ++i) {
        auto v = static_cast<NoiseVariant>(i);
        shaders[v] = std::make_unique<Shader>(
            std::string(SHADER_DIR) + "fullscreen.vert",
            std::string(SHADER_DIR) + "noise/noise" + std::to_string(i) + ".frag"
        );
    }

    blitShader = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "flow/blit.frag"
    );

    initTrail();

    std::uniform_real_distribution<float> dist(0.f, 1000.f);
    params.seed = dist(rng);
    params.time = 0.f;

    resetTrail();
}

void NoiseScene::update(float dt) {
    params.time += dt;

    params.color.r = up.r_level + up.r_amplitude * sinf(params.time * up.r_frequency + up.r_phase);
    params.color.g = up.g_level + up.g_amplitude * sinf(params.time * up.g_frequency + up.g_phase);
    params.color.b = up.b_level + up.b_amplitude * sinf(params.time * up.b_frequency + up.b_phase);

    params.trailDecay = up.td_level + up.td_amplitude * cosf(params.time * up.td_frequency + up.td_phase);
    params.speed      = up.speed_level + up.speed_amplitude * cosf(params.time * up.speed_frequency + up.speed_phase);
    params.intensity  = up.intensity_level + up.intensity_amplitude * cosf(params.time * up.intensity_frequency + up.intensity_phase);
}


void NoiseScene::render() {
    int src = ping;
    int dst = 1 - ping;

    glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[dst]);
    glViewport(0, 0, 1920, 1080);

    auto& sh = shaders[currentVariant];
    sh->use();

    sh->setFloat("u_seed", params.seed);
    sh->setFloat("u_time", params.time);
    sh->setVec2("u_resolution", 1920.f, 1080.f);
    sh->setVec3("u_color", params.color.r, params.color.g, params.color.b);
    sh->setFloat("u_scale", params.scale);
    sh->setInt("u_octaves", params.octaves);
    sh->setFloat("u_speed", params.speed);
    sh->setFloat("u_ampFactor", params.ampFactor);
    sh->setFloat("u_trailDecay", params.trailDecay);
    sh->setFloat("u_intensity", params.intensity);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, trailTex[src]);
    sh->setInt("u_trailTex", 0);

    quad->draw();

    // blit
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    blitShader->use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, trailTex[dst]);
    blitShader->setInt("u_tex", 0);
    quad->draw();

    ping = dst;
}


void NoiseScene::initTrail() {
    glGenFramebuffers(2, trailFBO);
    glGenTextures(2, trailTex);

    for (int i = 0; i < 2; ++i) {
        glBindTexture(GL_TEXTURE_2D, trailTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 1920, 1080, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, trailTex[i], 0);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void NoiseScene::resetTrail() {
    for (int i = 0; i < 2; ++i) {
        glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[i]);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    ping = 0;
    params.time = 0.f;

    std::uniform_real_distribution<float> dist(0.f, 1000.f);
    params.seed = dist(rng);
}


void NoiseScene::nextVariant() {
    int v = (static_cast<int>(currentVariant) + 1)
            % static_cast<int>(NoiseVariant::_Count);
    currentVariant = static_cast<NoiseVariant>(v);
    resetTrail();
}

void NoiseScene::prevVariant() {
    int v = (static_cast<int>(currentVariant) - 1
            + static_cast<int>(NoiseVariant::_Count))
            % static_cast<int>(NoiseVariant::_Count);
    currentVariant = static_cast<NoiseVariant>(v);
    resetTrail();
}

void NoiseScene::onKey(SDL_Keycode key) {
    if (key == SDLK_q) prevVariant();
    if (key == SDLK_e) nextVariant();
}


