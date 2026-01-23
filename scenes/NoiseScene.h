#pragma once
#include "../app/Scene.h"
#include "../gl/Shader.h"
#include "../gl/Quad.h"
#include <GL/glew.h>
#include <glm/vec3.hpp>
#include <memory>
#include <random>
#include <unordered_map>

enum class NoiseVariant {
    Noise1 = 0,
    Noise2,
    Noise3,
    Noise4,
    Noise5,
    Noise6,
    Noise7,
    Noise8,
    Noise9,
    Noise10,
    Noise11,
    Noise12,
    _Count
};

struct NoiseParams {
    float seed       = 0.0f;
    float time       = 0.0f;
    float scale      = 20.0f;
    int octaves      = 5;
    float ampFactor  = 0.50f;


    float speed      = 0.35f;
    float trailDecay = 1.0f;
    float intensity  = 0.1f;
    glm::vec3 color  = {0.0f, 0.0f, 0.0f};
};

struct updatingNoiseParams {
    //RGB
    float r_level       = 0.3f;
    float r_amplitude   = 0.1f;
    float r_frequency   = 0.3f;
    float r_phase       = 0.0f;

    float g_level       = 0.0f;
    float g_amplitude   = 0.1f;
    float g_frequency   = 0.5f;
    float g_phase       = 0.4f;

    float b_level       = 0.0f;
    float b_amplitude   = 0.1f;
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

class NoiseScene : public Scene {
    public:
        void onEnter() override;
        void update(float dt) override;
        void render() override;
        void onKey(SDL_Keycode key) override;

    private:
        void initTrail();
        void resetTrail();
        void nextVariant();
        void prevVariant();

        std::unique_ptr<Shader> blitShader;
        std::unique_ptr<Quad> quad;

        std::unordered_map<NoiseVariant, std::unique_ptr<Shader>> shaders;
        NoiseVariant currentVariant = NoiseVariant::Noise1;

        NoiseParams params;
        updatingNoiseParams up;

        GLuint trailFBO[2]{};
        GLuint trailTex[2]{};
        int ping = 0;

        std::mt19937 rng{std::random_device{}()};
};

