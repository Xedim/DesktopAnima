#include "FlowScene.h"
#include "../gl/Shader.h"
#include "../gl/Quad.h"
#include <GL/glew.h>
#include <random>

void FlowScene::onEnter() {
    flowShader = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "flow_color_change.frag"
    );

    blitShader = std::make_unique<Shader>(
        std::string(SHADER_DIR) + "fullscreen.vert",
        std::string(SHADER_DIR) + "blit.frag"
    );
    quad = std::make_unique<Quad>();
    initTrail(1920, 1080);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    params.seed = dist(gen);
    params.time = 0.0f;
}

void FlowScene::initTrail(int width, int height) {
    glGenFramebuffers(2, trailFBO);
    glGenTextures(2, trailTex);

    for (int i = 0; i < 2; ++i) {
        glBindTexture(GL_TEXTURE_2D, trailTex[i]);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB32F,
            width,
            height,
            0,
            GL_RGB,
            GL_FLOAT,
            nullptr
        );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[i]);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            trailTex[i],
            0
        );

        glClearColor(0,0,0,1);
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
    params.intensity  = up.intensity_level + up.intensity_amplitude * cosf(params.time * up.intensity_frequency
        + up.intensity_phase);
}

void FlowScene::render() {
    int src = ping;
    int dst = 1 - ping;

    glBindFramebuffer(GL_FRAMEBUFFER, trailFBO[dst]);
    glViewport(0, 0, 1920, 1080);

    flowShader->use();
    flowShader->setFloat("u_seed", params.seed);
    flowShader->setFloat("u_time", params.time);
    flowShader->setVec2("u_resolution", 1920.0f, 1080.0f);
    flowShader->setFloat("u_scale", params.scale);
    flowShader->setFloat("u_speed", params.speed);
    flowShader->setVec3(
        "u_color",
        params.color.r,
        params.color.g,
        params.color.b
    );
    flowShader->setFloat("u_trailDecay", params.trailDecay);
    flowShader->setFloat("u_intensity", params.intensity);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, trailTex[src]);
    flowShader->setInt("u_trailTex", 0);

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
