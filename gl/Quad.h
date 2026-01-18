#pragma once

class Quad {
public:
    Quad();
    ~Quad();

    void draw() const;

private:
    unsigned int vao = 0;
    unsigned int vbo = 0;
};
