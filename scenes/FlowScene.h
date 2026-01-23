#pragma once
#include "../app/Scene.h"
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <glm/vec3.hpp>
#include <memory>
#include <random>
#include <unordered_map>

class Shader;
class Quad;

enum class FlowVariant {
    Flow,
    Colorful,
    ColorChange,
    Space
};

struct FlowParams {
    float seed       = 0.0f;
    float time       = 0.0f;
    float scale      = 0.9f;

    float speed      = 2.0f;
    float trailDecay = 1.0f;
    float intensity  = 0.1f;
    glm::vec3 color  = {0.0f, 0.0f, 0.0f};
};

struct updatingFlowParams {
    //RGB
    float r_level       = 0.5f;
    float r_amplitude   = 0.5f;
    float r_frequency   = 0.3f;
    float r_phase       = 0.0f;

    float g_level       = 0.5f;
    float g_amplitude   = 0.5f;
    float g_frequency   = 0.5f;
    float g_phase       = 0.4f;

    float b_level       = 0.5f;
    float b_amplitude   = 0.5f;
    float b_frequency   = 0.5f;
    float b_phase       = 0.7f;

    //trailDecay
    float td_level     = 0.99f;
    float td_amplitude = 0.01f;
    float td_frequency = 0.06f;
    float td_phase     = 0.0f;

    // speed
    float speed_level     = 1.5f;
    float speed_amplitude = 1.0f;
    float speed_frequency = 0.012f;
    float speed_phase     = 0.0f;

    // intensity
    float intensity_level     = 0.20f;
    float intensity_amplitude = 0.02f;
    float intensity_frequency = 0.06f;
    float intensity_phase     = 0.0f;
};

class FlowScene : public Scene {
    public:
        std::mt19937 rng{std::random_device{}()};
        std::uniform_int_distribution<int> dist{0, 2};
        void onEnter() override;
        void update(float dt) override;
        void render() override;
        void initTrail(int width, int height);
        void resetTrail();
        void nextVariant();
        void prevVariant();
        void onKey(SDL_Keycode key) override
    ;

    private:
        std::unique_ptr<Shader> blitShader;
        std::unique_ptr<Quad> quad;

        FlowParams params;
        updatingFlowParams up;
        FlowVariant currentVariant = FlowVariant::Flow;
        std::unordered_map<FlowVariant, std::unique_ptr<Shader>> shaders;

        float time = 0.0f;
        float seed = 0.0f;
        GLuint trailFBO[2] = {0, 0};
        GLuint trailTex[2] = {0, 0};
        int ping = 0;
};