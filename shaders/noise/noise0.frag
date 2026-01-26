#version 330 core

in vec2 v_uv;
out vec4 fragColor;

// =======================================================
// CONFIG / UNIFORMS
// =======================================================

uniform float u_seed;
uniform float u_time;
uniform vec3  u_color;
uniform float u_scale;
uniform int   u_octaves;
uniform float u_speed;
uniform float u_ampFactor;

int u_pattern = 52;

// =======================================================
// MATH / OPERATORS
// =======================================================

float mNorm(mat2 m) {
    return length(vec4(m[0], m[1]));
}

mat2 rot(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

mat2 scale(float s) {
    return mat2(s, 0.0,
                0.0, s);
}

// =======================================================
// STATE INITIALIZATION
// =======================================================

mat2 initState(vec2 uv) {
    return mat2(
        (uv - 0.5) * u_scale,
        vec2(0.0, 1.0)
    );
}

// =======================================================
// MOTION / FLOW (STATE EVOLUTION)
// =======================================================

mat2 evolveState(mat2 P, vec2 uv, float t) {

    for (int i = 0; i < 10; i++) {
        P = scale(1.2) * P;
    }

    return P;
}

// =======================================================
// PATTERN FUNCTIONS (MEASUREMENT ONLY)
// =======================================================

float trigPattern(vec2 p, float t) {
    return sin(p.x * 10.0 + t) * cos(p.y * 10.0 + t);
}

float rationalPattern(vec2 p) {
    float x = length(p);
    return x * (1.0 - x) / (x * x + 0.1);
}

float weierstrass(vec2 p) {
    float s = 0.0;
    float a = 0.5;
    float b = 3.0;
    float x = length(p);

    for (int i = 0; i < 8; i++)
        s += pow(a, float(i)) *
             cos(pow(b, float(i)) * 3.1415 * x);

    return s;
}

// =======================================================
// PATTERN SELECTOR (OBSERVER)
// =======================================================


float fbm(vec2 x) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 6; i++) {
        v += a * sin(x.x + sin(x.y));
        x *= 2.0;
        a *= 0.5;
    }
    return v;
}

float evaluatePattern(mat2 P, float t) {

    vec2 p = P * vec2(1.0, 1.0);

    if (u_pattern == 0)  return trigPattern(p, t);
    if (u_pattern == 1)  return weierstrass(p);
    if (u_pattern == 2)  return rationalPattern(p);
    if (u_pattern == 3)  return sin(length(p) + t);
    if (u_pattern == 4)  return cos(p.x * p.y + t);
    if (u_pattern == 5)  return log(abs(p.x * p.y) + 1.0);

    // ===== ДОПОЛНИТЕЛЬНЫЕ ПАТТЕРНЫ =====

    // радиальные
    if (u_pattern == 6)  return sin(length(p) * 8.0 - t);
    if (u_pattern == 7)  return cos(length(p) * length(p) + t);
    if (u_pattern == 8)  return exp(-length(p)) * sin(10.0 * length(p));

    // угловые
    if (u_pattern == 9)  return sin(atan(p.y, p.x) * 6.0 + t);
    if (u_pattern == 10) return cos(atan(p.y, p.x) * 3.0 - t);

    // решётки / интерференция
    if (u_pattern == 11) return sin(p.x * 10.0) + cos(p.y * 10.0);
    if (u_pattern == 12) return sin(p.x * 10.0 + t) * sin(p.y * 10.0 - t);
    if (u_pattern == 13) return cos(p.x * p.x * 4.0) * sin(p.y * 4.0);

    // сингулярные
    if (u_pattern == 14) return 1.0 / (length(p) + 0.05);
    if (u_pattern == 15) return log(length(p) + 0.01) * sin(t);

    // хаотические
    if (u_pattern == 16) {
        float x = fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453);
        for (int i = 0; i < 4; i++)
            x = 3.9 * x * (1.0 - x);
        return x;
    }

    if (u_pattern == 17) {
        float x = fract(p.x + p.y);
        for (int i = 0; i < 5; i++)
            x = abs(1.0 - 2.0 * x);
        return x;
    }

    // квазипериодика
    if (u_pattern == 18) return sin(p.x * 7.0 + t) + sin(p.y * 11.0 - t);
    if (u_pattern == 19) return sin(p.x * 5.0) * cos(p.y * 13.0);

    // фрактальные отклики
    if (u_pattern == 20) {
        float x = length(p);
        float s = 0.0;
        for (int i = 0; i < 6; i++)
            s += sin(x * pow(2.0, float(i)) + t) / pow(2.0, float(i));
        return s;
    }

    if (u_pattern == 21) return fract(sin(p.x * 20.0) * sin(p.y * 20.0));

    // дискретные
    if (u_pattern == 22) return step(0.0, sin(p.x * 10.0 + t));
    if (u_pattern == 23) return step(p.x * p.y, 0.0);

    // волновые карты
    if (u_pattern == 24) return sin(p.x * p.y * 4.0 + t);
    if (u_pattern == 25) return cos(length(p + vec2(sin(t), cos(t))) * 6.0);

    // ===== WAVE MAPS =====

    // радиально-фазовые
    if (u_pattern == 26) return sin(length(p) * 6.0 + t);
    if (u_pattern == 27) return cos(length(p) * 8.0 - t);
    if (u_pattern == 28) return sin(length(p) * 4.0 + sin(t));
    if (u_pattern == 29) return sin(length(p) * 10.0) * cos(t);

    // спиральные
    if (u_pattern == 30) return sin(length(p) * 6.0 + atan(p.y, p.x) * 3.0 + t);
    if (u_pattern == 31) return cos(length(p) * 4.0 - atan(p.y, p.x) * 5.0 + t);
    if (u_pattern == 32) return sin(atan(p.y, p.x) * 8.0 + t);

    // линейные волны
    if (u_pattern == 33) return sin(p.x * 10.0 + t);
    if (u_pattern == 34) return cos(p.y * 10.0 - t);
    if (u_pattern == 35) return sin((p.x + p.y) * 6.0 + t);
    if (u_pattern == 36) return cos((p.x - p.y) * 6.0 - t);

    // интерференции
    if (u_pattern == 37) return sin(p.x * 6.0 + t) * cos(p.y * 6.0 - t);
    if (u_pattern == 38) return sin(p.x * 8.0 + t) + sin(p.y * 8.0 - t);
    if (u_pattern == 39) return cos(p.x * 5.0) * cos(p.y * 5.0 + t);

    // деформированные волны
    if (u_pattern == 40) return sin(p.x * p.y * 4.0 + t);
    if (u_pattern == 41) return cos((p.x * p.x + p.y * p.y) * 3.0 - t);
    if (u_pattern == 42) return sin((p.x + sin(t)) * 8.0) * cos((p.y + cos(t)) * 8.0);

    // затухающие
    if (u_pattern == 43) return exp(-length(p)) * sin(12.0 * length(p) + t);
    if (u_pattern == 44) return exp(-0.5 * length(p)) * cos(10.0 * length(p) - t);

    // фазово-смещённые
    if (u_pattern == 45) return sin(p.x * 6.0 + t) + cos(p.y * 6.0 + t * 0.7);

    // ===== FRACTAL PATTERNS =====

    // Mandelbrot-like (escape time)
    if (u_pattern == 46) {
        vec2 z = vec2(0.0);
        vec2 c = p * 0.6;
        float it = 0.0;

        for (int i = 0; i < 32; i++) {
            z = vec2(
                z.x*z.x - z.y*z.y,
                2.0*z.x*z.y
            ) + c;

            if (dot(z, z) > 4.0) break;
            it += 1.0;
        }
        return it / 32.0;
    }

    // Julia set (animated)
    if (u_pattern == 47) {
        vec2 z = p;
        vec2 c = vec2(0.4*cos(t), 0.4*sin(t));
        float it = 0.0;

        for (int i = 0; i < 32; i++) {
            z = vec2(
                z.x*z.x - z.y*z.y,
                2.0*z.x*z.y
            ) + c;

            if (dot(z, z) > 4.0) break;
            it += 1.0;
        }
        return it / 32.0;
    }

    // Box fold fractal
    if (u_pattern == 48) {
        vec2 z = p;
        float s = 0.0;

        for (int i = 0; i < 10; i++) {
            z = abs(z);
            z -= 0.5;
            z *= 1.5;
            s += exp(-length(z));
        }
        return s;
    }
    // Mirror fold
    if (u_pattern == 49) {
        vec2 z = p;
        for (int i = 0; i < 8; i++) {
            z = abs(z);
            z = z * 2.0 - 1.0;
        }
        return exp(-length(z));
    }

    if (u_pattern == 50) return fbm(p * 3.0);
    if (u_pattern == 51) return fbm(p * 3.0 + vec2(t));
    if (u_pattern == 52) {
        float s = 0.0;
        float a = 1.0;
        float f = 1.0;

        for (int i = 0; i < 7; i++) {
            s += a * sin(f * length(p) + t);
            f *= 2.0;
            a *= 0.5;
        }
        return s;
    }
    if (u_pattern == 53) {
        float s = 0.0;
        float a = 1.0;
        vec2 q = p;

        for (int i = 0; i < 6; i++) {
            s += a * sin(q.x + t) * cos(q.y - t);
            q *= 2.0;
            a *= 0.5;
        }
        return s;
    }
    // Orbit trap style
    if (u_pattern == 54) {
        vec2 z = p;
        float d = 10.0;

        for (int i = 0; i < 16; i++) {
            z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + p;
            d = min(d, length(z));
        }
        return exp(-d * 2.0);
    }
    if (u_pattern == 55) {
        float x = length(p);
        return sin(log(x + 0.01) * 6.0 + t);
    }
    if (u_pattern == 56) {
        float x = length(p);
        float s = 0.0;
        for (int i = 0; i < 6; i++) {
            s += sin(pow(x, float(i+1)) + t) / float(i+1);
        }
        return s;
    }
    if (u_pattern == 57) {
        vec2 q = fract(p * 3.0) - 0.5;
        float d = length(q);
        return exp(-10.0 * d);
    }
    if (u_pattern == 58) return sin(length(p) * 4.0 + fbm(p * 2.0) + t);
    if (u_pattern == 59) return cos(p.x * p.y + fbm(p + vec2(t)));
    if (u_pattern == 60) return fbm(p * 4.0 + vec2(sin(t), cos(t)));

    return 0.0;
}

// =======================================================
// FRACTAL ACCUMULATION
// =======================================================

float accumulate(mat2 P0, vec2 uv, float t) {

    float n   = 0.0;
    float amp = 1.0;
    mat2  P   = P0;

    for (int i = 0; i < u_octaves; i++) {
        n += evaluatePattern(P, t) * amp;
        P  = evolveState(P, uv, t);
        amp *= u_ampFactor;
    }

    return n;
}

// =======================================================
// OUTPUT / COLOR MAPPING
// =======================================================

vec4 mapColor(float n, float t) {
    vec3 color = u_color + vec3(n * 0.4) + vec3(0.2 + 0.05*sin(t), 0, 0);
    return vec4(color - 0.3, 1.0);
}

// =======================================================
// MAIN
// =======================================================

void main() {

    float t = u_time * u_speed;

    vec2 uv = v_uv;

    mat2 P0 = initState(uv);

    float n = accumulate(P0, uv, t);

    fragColor = mapColor(n, t);
}
