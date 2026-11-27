#pragma once

using i64 = long long;

using u32 = unsigned int;
using u64 = unsigned long long;

using f32 = float;
using f64 = double;

struct Vec4f {
  f32 x;
  f32 y;
  f32 z;
  f32 w;

  __host__ __device__
  Vec4f() {}

  __host__ __device__
  Vec4f(f32 x) : x(x), y(x), z(x), w(x) {}

  __host__ __device__
  Vec4f(f32 x, f32 y, f32 z, f32 w) : x(x), y(y), z(z), w(w) {}

  __host__ __device__
  operator float4() const {
    return {x, y, z, w};
  }

  __host__ __device__
  Vec4f& operator+=(const Vec4f& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    w += other.w;
    return *this;
  }

  __host__ __device__
  Vec4f& operator-=(const Vec4f& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    w -= other.w;
    return *this;
  }

  __host__ __device__
  Vec4f& operator*=(const f32& other) {
    x *= other;
    y *= other;
    z *= other;
    w *= other;
    return *this;
  }

  __host__ __device__
  Vec4f& operator/=(const f32& other) {
    x /= other;
    y /= other;
    z /= other;
    w /= other;
    return *this;
  }

  __host__ __device__ friend Vec4f operator+(Vec4f lhs, const Vec4f& rhs) { return lhs += rhs; }
  __host__ __device__ friend Vec4f operator-(Vec4f lhs, const Vec4f& rhs) { return lhs -= rhs; }

  __host__ __device__ friend Vec4f operator*(Vec4f lhs, const f32& rhs) { return lhs *= rhs; }
  __host__ __device__ friend Vec4f operator/(Vec4f lhs, const f32& rhs) { return lhs /= rhs; }

};