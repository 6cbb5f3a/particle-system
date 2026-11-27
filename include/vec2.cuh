#pragma once

using i64 = long long;

using u32 = unsigned int;
using u64 = unsigned long long;

using f32 = float;
using f64 = double;

struct Vec2f {
  f32 x;
  f32 y;

  __host__ __device__
  Vec2f() {}

  __host__ __device__
  Vec2f(f32 x) : x(x), y(x) {}

  __host__ __device__
  Vec2f(f32 x, f32 y) : x(x), y(y) {}

  __host__ __device__
  operator float2() const {
    return {x, y};
  }

  __host__ __device__
  Vec2f& operator+=(const Vec2f& other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  __host__ __device__
  Vec2f& operator-=(const Vec2f& other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }

  __host__ __device__
  Vec2f& operator*=(const f32& other) {
    x *= other;
    y *= other;
    return *this;
  }

  __host__ __device__
  Vec2f& operator/=(const f32& other) {
    x /= other;
    y /= other;
    return *this;
  }

  __host__ __device__ friend Vec2f operator+(Vec2f lhs, const Vec2f& rhs) { return lhs += rhs; }
  __host__ __device__ friend Vec2f operator-(Vec2f lhs, const Vec2f& rhs) { return lhs -= rhs; }

  __host__ __device__ friend Vec2f operator*(Vec2f lhs, const f32& rhs) { return lhs *= rhs; }
  __host__ __device__ friend Vec2f operator/(Vec2f lhs, const f32& rhs) { return lhs /= rhs; }

  __host__ __device__ friend Vec2f operator*(const f32& lhs, Vec2f rhs) { return rhs *= lhs; }
  __host__ __device__ friend Vec2f operator/(const f32& lhs, Vec2f rhs) { return rhs /= lhs; }

  __host__ __device__
  friend f32 dot(const Vec2f& a, const Vec2f& b) {
    return a.x * b.x + a.y * b.y;
  }
  __host__ __device__
  friend f32 cross(const Vec2f& a, const Vec2f& b) {
    return a.x * b.y - a.y * b.x;
  }
  __host__ __device__
  friend f32 length(const Vec2f& a) {
    return sqrt(dot(a, a));
  }
  __host__ __device__
  friend Vec2f normalize(const Vec2f& a) {
    return a * rsqrt(dot(a, a));
  }
};

struct Vec2i {
  int x;
  int y;

  __host__ __device__
  Vec2i() {}

  __host__ __device__
  Vec2i(int x, int y) : x(x), y(y) {}

  __host__ __device__
  Vec2i(const Vec2f& p) : x(p.x), y(p.y) {}

  __host__ __device__
  operator int2() const {
    return {x, y};
  }

  __host__ __device__
  Vec2i& operator+=(const Vec2i& other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  __host__ __device__
  Vec2i& operator-=(const Vec2i& other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }

  __host__ __device__ friend Vec2i operator+(Vec2i lhs, const Vec2i& rhs) { return lhs += rhs; }
  __host__ __device__ friend Vec2i operator-(Vec2i lhs, const Vec2i& rhs) { return lhs -= rhs; }
};

struct Vec2u {
  u32 x;
  u32 y;

  __host__ __device__
  Vec2u() {}

  __host__ __device__
  Vec2u(u32 x, u32 y) : x(x), y(y) {}

  __host__ __device__
  Vec2u(const Vec2f& p) : x(p.x), y(p.y) {}

  __host__ __device__
  operator uint2() const {
    return {x, y};
  }
};