#include "Shader.h"
#include <GL/glew.h>

#include <fstream>
#include <sstream>
#include <iostream>

static std::string loadFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << path << std::endl;
        return {};
    }

    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static void checkShader(unsigned int shader, const std::string& name) {
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "Shader compile error (" << name << "):\n" << log << std::endl;
    }
}

static void checkProgram(unsigned int program) {
    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetProgramInfoLog(program, 1024, nullptr, log);
        std::cerr << "Program link error:\n" << log << std::endl;
    }
}

Shader::Shader(const std::string& vertPath, const std::string& fragPath) {
    std::string vertSrc = loadFile(vertPath);
    std::string fragSrc = loadFile(fragPath);

    const char* v = vertSrc.c_str();
    const char* f = fragSrc.c_str();

    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &v, nullptr);
    glCompileShader(vs);
    checkShader(vs, vertPath);

    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &f, nullptr);
    glCompileShader(fs);
    checkShader(fs, fragPath);

    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    checkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);
}

Shader::~Shader() {
    if (program)
        glDeleteProgram(program);
}

void Shader::use() const {
    glUseProgram(program);
}

void Shader::setFloat(const std::string& name, float value) const {
    glUniform1f(glGetUniformLocation(program, name.c_str()), value);
}

void Shader::setInt(const std::string& name, int value) const {
    glUniform1i(glGetUniformLocation(program, name.c_str()), value);
}

void Shader::setVec2(const std::string& name, float x, float y) const {
    glUniform2f(glGetUniformLocation(program, name.c_str()), x, y);
}

void Shader::setVec3(const std::string& name, float x, float y, float z) const {
    glUniform3f(glGetUniformLocation(program, name.c_str()), x, y, z);
}
