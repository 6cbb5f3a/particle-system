#pragma once

using f32 = float;
using f64 = double;

struct Vec3f {
  f32 x;
  f32 y;
  f32 z;

  __host__ __device__
  Vec3f() {}

  __host__ __device__
  Vec3f(f32 x) : x(x), y(x), z(x) {}

  __host__ __device__
  Vec3f(f32 x, f32 y, f32 z, f32 w) : x(x), y(y), z(z) {}

  __host__ __device__
  operator float3() const {
    return {x, y, z};
  }

  __host__ __device__
  Vec3f& operator+=(const Vec3f& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }

  __host__ __device__
  Vec3f& operator-=(const Vec3f& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
  }

  __host__ __device__
  Vec3f& operator*=(const f32& other) {
    x *= other;
    y *= other;
    z *= other;
    return *this;
  }

  __host__ __device__
  Vec3f& operator/=(const f32& other) {
    x /= other;
    y /= other;
    z /= other;
    return *this;
  }

  __host__ __device__ friend Vec3f operator+(Vec3f lhs, const Vec3f& rhs) { return lhs += rhs; }
  __host__ __device__ friend Vec3f operator-(Vec3f lhs, const Vec3f& rhs) { return lhs -= rhs; }

  __host__ __device__ friend Vec3f operator*(Vec3f lhs, const f32& rhs) { return lhs *= rhs; }
  __host__ __device__ friend Vec3f operator/(Vec3f lhs, const f32& rhs) { return lhs /= rhs; }

};