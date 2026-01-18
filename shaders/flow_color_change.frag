#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform float u_seed;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_scale;
uniform float u_speed;
uniform vec3 u_color;
uniform float u_trailDecay;
uniform float u_intensity;

uniform sampler2D u_trailTex;

float hash(vec2 p) {
    return fract(sin(dot(p + u_seed, vec2(127.1,311.7))) * 43758.5453123);
}

vec2 flow(vec2 p) {
    float angle = hash(floor(p * u_scale)) * 6.2831853;
    return vec2(cos(angle), sin(angle));
}

void main() {
    vec2 uv_px = v_uv * u_resolution;

    vec2 f = flow(uv_px);
    vec2 offsetUV = uv_px - f * u_speed;
    offsetUV = mod(offsetUV, u_resolution);

    vec3 prevTrail = texture(u_trailTex, v_uv).rgb;
    vec3 offsetTrail = texture(u_trailTex, offsetUV / u_resolution).rgb;

    vec3 pixelColor = vec3(
        fract(sin(dot(uv_px + u_seed, vec2(12.9898,78.233))) * 43758.5453),
        fract(cos(dot(uv_px + u_seed, vec2(93.9898,67.345))) * 24634.6345),
        fract(sin(dot(uv_px + u_seed, vec2(54.5345,23.234))) * 56445.2345)
    );
    float inject = step(0.995, hash(uv_px + u_seed));
    vec3 newColor = offsetTrail * u_trailDecay + (pixelColor + u_color) * inject * u_intensity;
    vec3 color = mix(prevTrail, newColor, 0.5);
    fragColor = vec4(color, 1.0);
}
