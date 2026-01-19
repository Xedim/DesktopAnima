#version 330 core

in vec2 v_uv;
out vec4 fragColor;

// -----------------------
uniform float u_seed;      // сид
uniform float u_time;      // время
uniform vec3 u_color;      // основной цвет
uniform float u_scale;     // масштаб шума
uniform int u_octaves;     // число октав
uniform float u_speed;     // скорость анимации
uniform float u_ampFactor; // уменьшение амплитуды для каждой октавы
// -----------------------

float hash(vec2 p)
{
    return fract(sin(dot(p + u_seed, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f*f*(3.0-2.0*f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main()
{
    vec2 p = v_uv * u_scale;
    float t = u_time * u_speed;

    float n = 0.0;
    float amp = 1.0;

    for (int i = 0; i < u_octaves; i++)
    {
        vec2 offset = vec2(
            0,
            -2*t
        );
        n += noise(p + offset) * amp;
        p *= 2.0;
        amp *= u_ampFactor;
    }
    n = pow(n, 2);
    fragColor = vec4(n * u_color, 1.0);
}

//    shader->setVec2("u_resolution", 1920.0f, 1080.0f);
//    shader->setVec3("u_color", 0.8f, 0.0f, 0.0f);
//    shader->setFloat("u_scale", 20.0f);
//    shader->setInt("u_octaves", 5);
//    shader->setFloat("u_speed", 0.35f);
//    shader->setFloat("u_ampFactor", 0.30f);