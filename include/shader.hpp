#pragma once

#include <iostream>
#include <string>

#include <glad/gl.h>
#include <GLFW/glfw3.h>

class Shader {
public:
  Shader(const char* vert, const char* frag) : program(glCreateProgram()) {
    GLuint vertshader{glsl<GL_VERTEX_SHADER>(vert)};
    GLuint fragshader{glsl<GL_FRAGMENT_SHADER>(frag)};
    glAttachShader(program, vertshader);
    glAttachShader(program, fragshader);
    glLinkProgram(program);

    GLint size = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &size);
    if (size) {
      GLchar* log = new GLchar[size];
      glGetProgramInfoLog(program, size, nullptr, log);
      std::cerr << vert << " " << frag << std::endl;
      std::cerr << log << std::endl;
      delete[] log;
    }

    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success) {
      glDetachShader(program, vertshader);
      glDetachShader(program, fragshader);
      glDeleteShader(vertshader);
      glDeleteShader(fragshader);
    } else {
      std::cerr << __FILE__ << " " << __LINE__ << " " << __FUNCTION__ << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  Shader(const char* comp) : program(glCreateProgram()) {
    GLuint compshader{glsl<GL_COMPUTE_SHADER>(comp)};
    glAttachShader(program, compshader);
    glLinkProgram(program);

    GLint size = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &size);
    if (size) {
      GLchar* log = new GLchar[size];
      glGetProgramInfoLog(program, size, nullptr, log);
      std::cerr << comp << std::endl;
      std::cerr << log << std::endl;
      delete[] log;
    }

    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success) {
      glDetachShader(program, compshader);
      glDeleteShader(compshader);
    } else {
      std::cerr << __FILE__ << " " << __LINE__ << " " << __FUNCTION__ << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  GLuint operator()() const {
    return program;
  }
private:
  GLuint program;

  template <GLenum T> GLuint glsl(const char* filename) {
    std::string source;
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      file.open(filename);
      std::stringstream stream;
      stream << file.rdbuf();
      source = stream.str();
    } catch (std::ifstream::failure& e) {
      std::cerr << e.what() << std::endl;
      std::cerr << __FILE__ << " " << __LINE__ << " " << __FUNCTION__ << std::endl;
      exit(EXIT_FAILURE);
    }
    const GLchar* code = source.c_str();
    GLuint shader = glCreateShader(T);
    glShaderSource(shader, 1, &code, nullptr);
    glCompileShader(shader);

    GLint size = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &size);
    if (size) {
      GLchar* log = new GLchar[size];
      glGetShaderInfoLog(shader, size, nullptr, log);
      std::cerr << filename << std::endl;
      std::cerr << log << std::endl;
      delete[] log;
    }

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success) {
      return shader;
    } else {
      std::cerr << __FILE__ << " " << __LINE__ << " " << __FUNCTION__ << std::endl;
      exit(EXIT_FAILURE);
    }
  }
};
