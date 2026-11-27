#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <iomanip>
#include <algorithm>
#include <set>
#include <map>
#include <numeric>
#include <random>
#include <ranges>
#include <chrono>
#include <cmath>

// glad
#include <glad/gl.h>

// glfw
#include <GLFW/glfw3.h>

// cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// imgui
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "shader.hpp"

#include "vec2.cuh"
#include "vec3.cuh"
#include "vec4.cuh"

using i64 = long long;

using u32 = unsigned int;
using u64 = unsigned long long;

using f32 = float;
using f64 = double;

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());


struct Vertex {
  Vec2f pos;
  Vec2f uv;
};

struct Particles {
  u32 count = 0;

  Vec2f* pos = nullptr;
  Vec2f* vel = nullptr;
  Vec2f* dir = nullptr;

  u32* key = nullptr;
  u32* ord = nullptr;
};

struct World {
  Particles particles;

  u32* beg = nullptr;
  u32* end = nullptr;

  Vec2f* sum_pos = nullptr;
  Vec2f* sum_vel = nullptr;
};

constexpr u32 MAX_PARTICLES = (1 << 20);
constexpr f32 ZOOM_MAX = 1.0f;
constexpr f32 ZOOM_MIN = 0.05f;

constexpr int window_w = 16 * 64;
constexpr int window_h = 10 * 64;

f32 zoom = 1.0f;

f32 offset_x = 0.0f;
f32 offset_y = 0.0f;

int render_w = window_w;
int render_h = window_h;

World world;
World new_world;

__constant__ constexpr int world_w = 128;
__constant__ constexpr int world_h = 128;

__constant__ constexpr f32 max_speed = 0.3f;
__constant__ constexpr f32 max_force = 0.01f;

__constant__ int d_render_w;
__constant__ int d_render_h;

__host__ __device__
u32 spread(u32 n) {
  n = (n | (n << 8)) & 0x00FF00FF;
  n = (n | (n << 4)) & 0x0F0F0F0F;
  n = (n | (n << 2)) & 0x33333333;
  n = (n | (n << 1)) & 0x55555555;
  return n;
}

__host__ __device__
u32 z_order(u32 x, u32 y) {
  return spread(x) | (spread(y) << 1);
}

__host__ __device__
f32 fractf(f32 x) {
  return x - floorf(x);
}

__global__
void grid_hash(Particles particles) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < particles.count) {
    u32 grid_x = u32(particles.pos[x].x);
    u32 grid_y = u32(particles.pos[x].y);
    particles.key[x] = z_order(grid_x, grid_y);
    particles.ord[x] = x;
  }
}

__global__
void reorder(Particles particles, Particles sorted) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < particles.count) {
    u32 y = particles.ord[x];
    sorted.pos[x] = particles.pos[y];
    sorted.vel[x] = particles.vel[y];
    sorted.dir[x] = particles.dir[y];
    sorted.ord[x] = x;
  }
}

__global__
void compute_bounds(World world) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int r = world.particles.count - 1;
  if (x <= r) {
    u32 key = world.particles.key[x];
    if (x == 0 || key != world.particles.key[x - 1]) {
      world.beg[key] = x;
    }
    if (x == r || key != world.particles.key[x + 1]) {
      world.end[key] = x + 1;
    }
  }
}

__global__
void update(World world) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < world_w && y < world_h) {
    Vec2f sum_pos = Vec2f(0.0f, 0.0f);
    Vec2f sum_vel = Vec2f(0.0f, 0.0f);
    u32 key = z_order(x, y);
    u32 beg = world.beg[key];
    u32 end = world.end[key];
    for (u32 i = beg; i < end; ++i) {
      Vec2f pos = world.particles.pos[i];
      Vec2f vel = world.particles.vel[i];
      sum_pos += pos;
      sum_vel += vel;
    }
    world.sum_pos[key] = sum_pos;
    world.sum_vel[key] = sum_vel;
  }
}

__global__
void process(
  World world,
  World new_world,
  f32 ali_weight,
  f32 coh_weight,
  f32 sep_weight
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < world.particles.count) {
    Vec2f pos = world.particles.pos[x];
    Vec2f vel = world.particles.vel[x];
    Vec2f dir = world.particles.dir[x];
    Vec2u grid = pos;

    f32 eps = 1e-5f;
    auto safe_normalize = [](Vec2f p) {
      f32 len = length(p);
      if (len > 0.0f) {
        return p / len;
      } else {
        return p;
      }
    };

    auto limit = [](Vec2f p, f32 max) {
      f32 len = length(p);
      if (len > max) {
        return p / len * max;
      } else {
        return p;
      }
    };

    int neighbor_count = -1;

    Vec2f coh = {0.0f, 0.0f};
    Vec2f sep = {0.0f, 0.0f};
    Vec2f ali = {
      -vel.x,
      -vel.y
    };

    for (int dy = -2; dy <= 2; ++dy) {
      for (int dx = -2; dx <= 2; ++dx) {
        u32 key = z_order(
          (grid.x + dx) & (world_w - 1),
          (grid.y + dy) & (world_h - 1)
        );
        u32 beg = world.beg[key];
        u32 end = world.end[key];
        for (u32 i = beg; i < end; ++i) {
          u32 ord = world.particles.ord[i];
          if (ord == x) {
            continue;
          }
          Vec2f other_pos = world.particles.pos[ord];
          Vec2f delta = other_pos - pos;
          delta.x -= world_w * floorf(delta.x / f32(world_w) + 0.5f);
          delta.y -= world_h * floorf(delta.y / f32(world_h) + 0.5f);

          f32 len = length(delta) + eps;
          if (len < 1.5f) {
            sep -= delta / (len * len);
          }
        }
      }
    }
    for (int dy = -4; dy <= 4; ++dy) {
      for (int dx = -4; dx <= 4; ++dx) {
        u32 key = z_order(
          (grid.x + dx) & (world_w - 1),
          (grid.y + dy) & (world_h - 1)
        );
        u32 beg = world.beg[key];
        u32 end = world.end[key];

        u32 count = end - beg;
        if (count == 0) {
          continue;
        }
        Vec2f avg_pos = world.sum_pos[key] / f32(count);
        Vec2f avg_vel = world.sum_vel[key] / f32(count);

        Vec2f delta = avg_pos - pos;
        delta.x -= world_w * floorf(delta.x / f32(world_w) + 0.5f);
        delta.y -= world_h * floorf(delta.y / f32(world_h) + 0.5f);

        f32 len = length(delta) + eps;
        if (len < 4.0f) {
          ali += count * avg_vel;
          coh += count * delta;
          neighbor_count += count;
        }
      }
    }

    if (neighbor_count > 0) {
      ali = safe_normalize(ali / f32(neighbor_count)) * max_speed;
      coh = safe_normalize(coh / f32(neighbor_count)) * max_speed;
      ali = limit(ali - vel, max_force);
      coh = limit(coh - vel, max_force);
      sep = limit(sep - vel, max_force);

      vel += ali_weight * ali
           + coh_weight * coh
           + sep_weight * sep;
    }

    new_world.particles.pos[x] = pos;
    new_world.particles.vel[x] = vel;
    new_world.particles.dir[x] = dir;
    if (dot(vel, vel) > eps) {
      new_world.particles.dir[x] = normalize(vel);
      new_world.particles.pos[x] = {
        fmod(pos.x + vel.x + world_w, f32(world_w)),
        fmod(pos.y + vel.y + world_h, f32(world_h))
      };
    }
  }
}

__global__
void shade_pixel(
  cudaSurfaceObject_t surface,
  World world,
  f32 zoom,
  f32 offset_x,
  f32 offset_y
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < d_render_w && y < d_render_h) {

    f32 scale = zoom * min(
      f32(world_w) / d_render_w,
      f32(world_h) / d_render_h
    );

    Vec2f pixel = {
      fmod(f32(x + 0.5f) * scale + offset_x, f32(world_w)),
      fmod(f32(y + 0.5f) * scale + offset_y, f32(world_h))
    };

    int grid_x = int(pixel.x);
    int grid_y = int(pixel.y);

    Vec4f color = (grid_x + grid_y) % 2 == 0
        ? Vec4f(0.2f, 0.2f, 0.2f, 1.0f)
        : Vec4f(0.1f, 0.1f, 0.1f, 1.0f);

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        u32 key = z_order(
          (grid_x + dx) & (world_w - 1),
          (grid_y + dy) & (world_h - 1)
        );
        u32 beg = world.beg[key];
        u32 end = world.end[key];
        for (u32 i = beg; i < end; ++i) {
          Vec2f pos = world.particles.pos[i];
          Vec2f dir = world.particles.dir[i];
          Vec2f delta = pixel - pos;

          delta.x -= world_w * floorf(delta.x / f32(world_w) + 0.5f);
          delta.y -= world_h * floorf(delta.y / f32(world_h) + 0.5f);

          Vec2f right = {
            -dir.y,
            +dir.x
          };
          f32 y = dot(delta, dir);
          f32 x = dot(delta, right);

          f32 height = 0.6f;
          if (y >= -height * 0.5f && y <= height * 0.5f) {
            f32 t = (y + height * 0.5f) / height;
            if (abs(x) < (1.0f - t) * 0.2f) {
              color = {
                0.4, 0.7, 0.3, 1.0f
              };
            }
          }
        }
      }
    }

    surf2Dwrite(static_cast<float4>(color), surface, x * sizeof(float4), y);
  }
}

int main(int argc, char* argv[]) {
  {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  }

  GLFWwindow* window = glfwCreateWindow(window_w, window_h, "CUDA", nullptr, nullptr);
  glfwGetFramebufferSize(window, &render_w, &render_h);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  gladLoadGL(glfwGetProcAddress);

  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  {
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.IniFilename = nullptr;
    io.Fonts->AddFontFromFileTTF("assets/fonts/NotoMono-Regular.ttf", 16.0f);
  }
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 460");
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "==================OpenGL=================" << std::endl;
    std::cout << "OpenGL Renderer:         " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "OpenGL Version:          " << glGetString(GL_VERSION) << std::endl;
    std::cout << "===================CUDA==================" << std::endl;
    std::cout << "Device:                  " << prop.name << std::endl;
    std::cout << "Compute Capability:      " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multiprocessors (SM):    " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Shared Mem / Block:  " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Total Shared Mem / SM:   " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
    std::cout << "Max Threads / Block:     " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Blocks / SM:         " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Regs / Block:            " << prop.regsPerBlock << std::endl;
    std::cout << "Memory Bus Width:        " << prop.memoryBusWidth << " bit" << std::endl;
    std::cout << "L2 Cache Size:           " << prop.l2CacheSize / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "=========================================" << std::endl;
  }
  Vertex vertices[] = {
    {{-1.0f, -1.0f}, {0.0f, 0.0f}},
    {{ 1.0f, -1.0f}, {1.0f, 0.0f}},
    {{ 1.0f,  1.0f}, {1.0f, 1.0f}},
    {{-1.0f,  1.0f}, {0.0f, 1.0f}}
  };

  GLuint vertex_array;
  GLuint vertex_buffer;

  glCreateVertexArrays(1, &vertex_array);
  glCreateBuffers(1, &vertex_buffer);

  glNamedBufferStorage(vertex_buffer, sizeof(vertices), vertices, 0);
  glVertexArrayVertexBuffer(vertex_array, 0, vertex_buffer, 0, sizeof(Vertex));

  glEnableVertexArrayAttrib(vertex_array, 0);
  glVertexArrayAttribFormat(vertex_array, 0, 2, GL_FLOAT, GL_FALSE, offsetof(Vertex, pos));
  glVertexArrayAttribBinding(vertex_array, 0, 0);

  glEnableVertexArrayAttrib(vertex_array, 1);
  glVertexArrayAttribFormat(vertex_array, 1, 2, GL_FLOAT, GL_FALSE, offsetof(Vertex, uv));
  glVertexArrayAttribBinding(vertex_array, 1, 0);


  GLuint texture;
  glCreateTextures(GL_TEXTURE_2D, 1, &texture);
  glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTextureStorage2D(texture, 1, GL_RGBA32F, render_w, render_h);

  cudaGraphicsResource_t cuda_tex_res;
  cudaGraphicsGLRegisterImage(
    &cuda_tex_res,
    texture,
    GL_TEXTURE_2D,
    cudaGraphicsRegisterFlagsSurfaceLoadStore
  );

  f64 delta = 0.0;
  f64 accum = 0.0;

  Shader shader("shaders/vert.glsl", "shaders/frag.glsl");

  {
    cudaMemcpyToSymbol(d_render_w, &render_w, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_render_h, &render_h, sizeof(int), 0, cudaMemcpyHostToDevice);

    {
      cudaMalloc(&new_world.beg, world_w * world_h * sizeof(u32));
      cudaMalloc(&new_world.end, world_w * world_h * sizeof(u32));
      cudaMalloc(&new_world.sum_pos, world_w * world_h * sizeof(Vec2f));
      cudaMalloc(&new_world.sum_vel, world_w * world_h * sizeof(Vec2f));

      cudaMalloc(&new_world.particles.pos, MAX_PARTICLES * sizeof(Vec2f));
      cudaMalloc(&new_world.particles.vel, MAX_PARTICLES * sizeof(Vec2f));
      cudaMalloc(&new_world.particles.dir, MAX_PARTICLES * sizeof(Vec2f));
      cudaMalloc(&new_world.particles.key, MAX_PARTICLES * sizeof(u32));
      cudaMalloc(&new_world.particles.ord, MAX_PARTICLES * sizeof(u32));
    }

    {
      cudaMalloc(&world.beg, world_w * world_h * sizeof(u32));
      cudaMalloc(&world.end, world_w * world_h * sizeof(u32));
      cudaMalloc(&world.sum_pos, world_w * world_h * sizeof(Vec2f));
      cudaMalloc(&world.sum_vel, world_w * world_h * sizeof(Vec2f));

      cudaMalloc(&world.particles.pos, MAX_PARTICLES * sizeof(Vec2f));
      cudaMalloc(&world.particles.vel, MAX_PARTICLES * sizeof(Vec2f));
      cudaMalloc(&world.particles.dir, MAX_PARTICLES * sizeof(Vec2f));
      cudaMalloc(&world.particles.key, MAX_PARTICLES * sizeof(u32));
      cudaMalloc(&world.particles.ord, MAX_PARTICLES * sizeof(u32));
    }
    cudaMemset(world.beg, 0xff, world_w * world_h * sizeof(u32));
    cudaMemset(world.end, 0xff, world_w * world_h * sizeof(u32));
    cudaMemset(world.sum_pos, 0x00, world_w * world_h * sizeof(Vec2f));
    cudaMemset(world.sum_vel, 0x00, world_w * world_h * sizeof(Vec2f));

    world.particles.count = (1 << 12);
    std::vector<Vec2f> pos(world.particles.count);
    std::vector<Vec2f> vel(world.particles.count);
    std::vector<Vec2f> dir(world.particles.count);

    std::vector<u32> key(world.particles.count);
    std::vector<u32> ord(world.particles.count);
    for (int i = 0; i < size(pos); ++i) {
      pos[i] = {
        std::uniform_real_distribution<f32>(0.0f, f32(world_w))(rng),
        std::uniform_real_distribution<f32>(0.0f, f32(world_h))(rng)
      };
      do {
        vel[i] = {
          std::uniform_real_distribution<f32>(-max_speed, max_speed)(rng),
          std::uniform_real_distribution<f32>(-max_speed, max_speed)(rng)
        };
      } while (dot(vel[i], vel[i]) > max_speed * max_speed);

      dir[i] = normalize(vel[i]);
      key[i] = z_order(pos[i].x, pos[i].y);
      ord[i] = i;
    }

    cudaMemcpy(world.particles.pos, pos.data(), pos.size() * sizeof(Vec2f), cudaMemcpyHostToDevice);
    cudaMemcpy(world.particles.vel, vel.data(), vel.size() * sizeof(Vec2f), cudaMemcpyHostToDevice);
    cudaMemcpy(world.particles.dir, dir.data(), dir.size() * sizeof(Vec2f), cudaMemcpyHostToDevice);
    cudaMemcpy(world.particles.key, key.data(), key.size() * sizeof(u32), cudaMemcpyHostToDevice);
    cudaMemcpy(world.particles.ord, ord.data(), ord.size() * sizeof(u32), cudaMemcpyHostToDevice);
  }

  while (!glfwWindowShouldClose(window)) {
    static f32 ali_weight = 0.10f;
    static f32 coh_weight = 0.12f;
    static f32 sep_weight = 0.18f;

    f64 begin = glfwGetTime();

    glfwPollEvents();
    { // ImGui
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      ImGuiViewport* viewport = ImGui::GetMainViewport();
      { // Settings
        ImGui::SetNextWindowSize(
          ImVec2(viewport->Size.x * 0.30, viewport->Size.y * 0.25),
          ImGuiCond_Once
        );
        ImGui::SetNextWindowPos(
          ImVec2(viewport->Size.x * 0.70, viewport->Size.y * 0.00),
          ImGuiCond_Once
        );
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize
                               | ImGuiWindowFlags_NoMove
                               | ImGuiWindowFlags_NoCollapse;
        ImGui::Begin("Settings", nullptr, flags);
        ImGui::SliderFloat("Ali", &ali_weight, 0.0f, 0.2f);
        ImGui::SliderFloat("Coh", &coh_weight, 0.0f, 0.2f);
        ImGui::SliderFloat("Sep", &sep_weight, 0.0f, 0.2f);
        ImGui::End();
      }
      { // Debug
        ImGui::SetNextWindowSize(
          ImVec2(viewport->Size.x * 0.30, viewport->Size.y * 0.25),
          ImGuiCond_Once
        );
        ImGui::SetNextWindowPos(
          ImVec2(viewport->Size.x * 0.70, viewport->Size.y * 0.25),
          ImGuiCond_Once
        );
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize
                               | ImGuiWindowFlags_NoMove
                               | ImGuiWindowFlags_NoCollapse;
        ImGui::Begin("Debug", nullptr, flags);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Particles: %d", world.particles.count);
        ImGui::Text("Size: %d x %d", world_w, world_h);
        ImGui::Text("Zoom: %.2f", zoom);
        ImGui::Text("Offset: %.2f %.2f", offset_x, offset_y);
        ImGui::End();
      }

      if (!io.WantCaptureMouse) {
        if (io.MouseWheel != 0.0f) {
          ImVec2 mouse_pos = io.MousePos;
          f32 pixel_x = (f32(render_w) / window_w) * (0.5 + mouse_pos.x);
          f32 pixel_y = (f32(render_h) / window_h) * (0.5 - mouse_pos.y + window_h);

          f32 base_scale = std::min(
            f32(world_w) / render_w,
            f32(world_h) / render_h
          );

          offset_x += zoom * base_scale * pixel_x;
          offset_y += zoom * base_scale * pixel_y;

          zoom = std::clamp<f32>(zoom * std::pow(0.9, io.MouseWheel), ZOOM_MIN, ZOOM_MAX);

          offset_x -= zoom * base_scale * pixel_x;
          offset_y -= zoom * base_scale * pixel_y;

          offset_x = std::fmod(std::fmod(offset_x, world_w) + world_w, world_w);
          offset_y = std::fmod(std::fmod(offset_y, world_h) + world_h, world_h);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          ImVec2 mouse_pos = io.MousePos;
          f32 pixel_x = (f32(render_w) / window_w) * (0.5 + mouse_pos.x);
          f32 pixel_y = (f32(render_h) / window_h) * (0.5 - mouse_pos.y + window_h);

          f32 scale = zoom * std::min(
            f32(world_w) / render_w,
            f32(world_h) / render_h
          );

          int world_x = int(scale * pixel_x + offset_x) & (world_w - 1);
          int world_y = int(scale * pixel_y + offset_y) & (world_h - 1);

          Vec2f pos = {
            std::fmod(scale * pixel_x + offset_x, f32(world_w)),
            std::fmod(scale * pixel_y + offset_y, f32(world_h))
          };
          Vec2f vel = {
            0.0f, 0.0f
          };
          {
            f32 radius = max_speed;
            std::uniform_real_distribution<f32> dist(-radius, radius);
            do {
              vel = {
                dist(rng),
                dist(rng)
              };
            } while (dot(vel, vel) > radius * radius);
          }
          Vec2f dir = normalize(vel);

          world.particles.count = min(world.particles.count + 1, MAX_PARTICLES);
          cudaMemcpy(world.particles.pos + world.particles.count - 1, &pos, sizeof(Vec2f), cudaMemcpyHostToDevice);
          cudaMemcpy(world.particles.vel + world.particles.count - 1, &vel, sizeof(Vec2f), cudaMemcpyHostToDevice);
          cudaMemcpy(world.particles.dir + world.particles.count - 1, &dir, sizeof(Vec2f), cudaMemcpyHostToDevice);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
          ImVec2 delta = io.MouseDelta;
          f32 scale = zoom * std::min(
            f32(world_w) / render_w,
            f32(world_h) / render_h
          );
          offset_x -= delta.x * scale * (f32(render_w) / window_w);
          offset_y += delta.y * scale * (f32(render_h) / window_h);

          offset_x = std::fmod(std::fmod(offset_x, world_w) + world_w, world_w);
          offset_y = std::fmod(std::fmod(offset_y, world_h) + world_h, world_h);
        }
      }
      if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      }
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) != GLFW_PRESS) {
      {
        dim3 block(32, 8);
        dim3 grid(
          (world_w + block.x - 1) / block.x,
          (world_h + block.y - 1) / block.y
        );
        update<<<grid, block>>>(
          world
        );
      }
      {
        u32 block = 256;
        u32 grid = max(block, (world.particles.count + block - 1)) / block;
        process<<<grid, block>>>(
          world,
          new_world,
          ali_weight,
          coh_weight,
          sep_weight
        );
        std::swap(world.particles.pos, new_world.particles.pos);
        std::swap(world.particles.vel, new_world.particles.vel);
        std::swap(world.particles.dir, new_world.particles.dir);
      }
    }
    {
      u32 block = 256;
      u32 grid = max(block, (world.particles.count + block - 1)) / block;


      grid_hash<<<grid, block>>>(
        world.particles
      );

      thrust::device_ptr<u32> key(world.particles.key);
      thrust::device_ptr<u32> ord(world.particles.ord);
      thrust::sort_by_key(key, key + world.particles.count, ord);

      reorder<<<grid, block>>>(
        world.particles,
        new_world.particles
      );

      std::swap(world.particles.pos, new_world.particles.pos);
      std::swap(world.particles.vel, new_world.particles.vel);
      std::swap(world.particles.dir, new_world.particles.dir);
      std::swap(world.particles.ord, new_world.particles.ord);

      cudaMemset(world.beg, 0xff, world_w * world_h * sizeof(u32));
      cudaMemset(world.end, 0xff, world_w * world_h * sizeof(u32));
      compute_bounds<<<grid, block>>>(
        world
      );
    }
    {
      cudaGraphicsMapResources(1, &cuda_tex_res, 0);
      cudaArray_t cuda_tex;
      cudaGraphicsSubResourceGetMappedArray(&cuda_tex, cuda_tex_res, 0, 0);

      cudaResourceDesc desc = {};
      desc.resType = cudaResourceTypeArray;
      desc.res.array.array = cuda_tex;

      cudaSurfaceObject_t surface;
      cudaCreateSurfaceObject(&surface, &desc);
      dim3 block(32, 8);
      dim3 grid(
        (render_w + block.x - 1) / block.x,
        (render_h + block.y - 1) / block.y
      );
      shade_pixel<<<grid, block>>>(
        surface,
        world,
        zoom,
        offset_x,
        offset_y
      );
      cudaDestroySurfaceObject(surface);
      cudaGraphicsUnmapResources(1, &cuda_tex_res, 0);
    }

    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    {
      glViewport(0, 0, render_w, render_h);
      glUseProgram(shader());
      glBindTextureUnit(0, texture);
      glBindVertexArray(vertex_array);
      glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
    {
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    glfwSwapBuffers(window);

    delta = glfwGetTime() - begin;
    accum = accum + delta;
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
