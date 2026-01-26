#version 400 core

in vec2 v_uv;
out vec4 fragColor;

// ==================== CONFIG ====================
uniform float u_time;
const vec2 u_center = vec2(-0.7436438870371587, 0.13182590420531198);
const float u_speed = -10.2;  // скорость приближения

// ==================== MANDELBROT ====================
float mandelbrotIter(vec2 c)
{
    vec2 z = vec2(0.0);
    vec2 dz = vec2(1.0, 0.0); // для гладкого подсчета
    float it = 0.0;
    const int maxIter = 512;
    const float escapeRadius = 16.0;

    for(int i = 0; i < maxIter; i++)
    {
        // стандартное итерационное соотношение
        vec2 z_new = vec2(
            z.x*z.x - z.y*z.y + c.x,
            2.0*z.x*z.y + c.y
        );
        z = z_new;

        if(dot(z,z) > escapeRadius) break;
        it += 1.0;
    }

    // плавное значение итераций (не используем ключевое слово smooth)
    float log_zn = log(dot(z,z)) / 2.0;
    float nu = log(log_zn / log(2.0)) / log(2.0);
    float smoothIter = it + 1.0 - nu;

    return smoothIter / float(maxIter);
}

// ==================== COLORING ====================
vec3 colorPalette(float t)
{
    // t ∈ [0,1], плавный градиент через синусы
    return vec3(
        0.5 + 0.5 * cos(3.0 + 6.2831 * t),
        0.5 + 0.5 * cos(1.0 + 6.2831 * t),
        0.5 + 0.5 * cos(2.0 + 6.2831 * t)
    );
}

// ==================== MAIN ====================
void main()
{
    // Экспоненциальный zoom для бесконечного приближения
    float zoom = pow(1.02, u_time * u_speed);

    // Координаты с центром
    vec2 uv = (v_uv - 0.5) * zoom + u_center;

    // Получаем нормализованное число итераций
    float iter = mandelbrotIter(uv);

    // Получаем цвет по градиенту
    vec3 color = colorPalette(iter);

    fragColor = vec4(color, 1.0);
}
