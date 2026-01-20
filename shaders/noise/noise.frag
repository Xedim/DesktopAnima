#version 330 core

in vec2 v_uv;
out vec4 fragColor;

// -----------------------
uniform float u_seed;      // сид
uniform float u_time;      // время
uniform vec3 u_color;      // основной цвет
uniform float u_scale;     // масштаб координат
uniform int u_octaves;     // число октав
uniform float u_speed;     // скорость анимации
uniform float u_ampFactor; // уменьшение амплитуды для каждой октавы
//uniform int u_pattern;     // выбор паттерна
int u_pattern = 3;
// -----------------------

// --- хэш / базовый шум ---
float hash(vec2 p) {
    return fract(sin(dot(p + u_seed, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f*f*(3.0-2.0*f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// --- Паттерны ---
float polyGradient(float x) { return x*x*(3.0-2.0*x); }
float rationalPattern(float x) { return x*(1.0-x)/(x*x+0.1); }
float trigWave(vec2 uv, float t) { return sin(uv.x*10.0+t)*cos(uv.y*10.0+t); }
float hyperWarp(float x) { return tanh(x*3.0); }
float logisticMap(float x, float r, int n) { for(int i=0;i<n;i++) x=r*x*(1.0-x); return x; }
float weierstrassPattern(float x) { float sum=0.0; float a=0.5; float b=3.0; for(int n=0;n<10;n++) sum+=pow(a,float(n))*cos(pow(b,float(n))*3.1415*x); return sum; }
float cantorPattern(float x) { float y=0.0; float factor=1.0; for(int i=0;i<16;i++){ if(x<1.0/3.0){ x*=3.0; } else if(x>2.0/3.0){ y+=factor; x=3.0*x-2.0; } else return y+factor*0.5; factor*=0.5; } return y; }
float smoothStepFunc(float x){ float eps=0.01; return 0.5*(1.0+tanh(x/eps)); }
float xPowY(float x, float y){ return exp(y*log(x)); }
float tentMap(float x){ return x<0.5 ? 2.0*x : 2.0*(1.0-x); }
float logRamp(float x){ return log(1.0+x); }
float expRamp(float x){ return exp(x)-1.0; }
vec2 warpUV(vec2 uv, float t){ uv+=vec2(sin(uv.y*10.0+t),cos(uv.x*10.0+t))*0.05; return uv; }
float polyWave(float x){ return x*x*(1.0-x)*(1.0-x); }
float rationalWarp(float x){ return x/(x+0.5); }

// --- Выбор паттерна ---
float patternValue(vec2 uv, float t) {
    float x = uv.x;
    float y = uv.y;
    if(u_pattern == 0) return trigWave(uv,t);
    else if(u_pattern == 1) return weierstrassPattern(x);
    else if(u_pattern == 2) return noise(uv*10.0 + t);
    else if(u_pattern == 3) return weierstrassPattern(xPowY(t, x));
    return 0.0;
}

// --- Главная функция ---
void main() {
    vec2 p = v_uv * u_scale;
    float t = u_time * u_speed;

    float n = 0.0;
    float amp = 1.0;

    for (int i = 0; i < u_octaves; i++) {
        vec2 offset = vec2(-10, -2.0*t);
        n += patternValue(p + offset, t) * amp;
        p += warpUV(p, t) * 0.2; // небольшой warp для движения
        amp *= u_ampFactor;
    }

    vec3 color = u_color;
    color.r += n * 0.5;
    color.g += n * 0.0;
    color.b += n * 0.0;
    color += vec3(n*0.5); // применяем ко всем каналам
    fragColor = vec4(color - 0.5, 1.0); // сдвиг цвета для визуализации
}
