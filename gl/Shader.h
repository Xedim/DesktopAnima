#pragma once
#include <string>

class Shader {
public:
    Shader(const std::string& vertPath, const std::string& fragPath);
    ~Shader();

    void use() const;

    void setFloat(const std::string& name, float value) const;
    void setInt(const std::string& name, int value) const;
    void setVec2(const std::string& name, float x, float y) const;
    void setVec3(const std::string& name, float x, float y, float z) const;

private:
    unsigned int program;
};

