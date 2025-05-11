#pragma once
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

struct vec2d {
	double x, y;

	__host__ __device__ vec2d() : x(0), y(0) {}
	__host__ __device__ vec2d(double x, double y) : x(x), y(y) {}

	__host__ __device__ vec2d operator+(const vec2d& b) const {
		return vec2d(x + b.x, y + b.y);
	}
	__host__ __device__ vec2d operator-(const vec2d& b) const {
		return vec2d(x - b.x, y - b.y);
	}
	__host__ __device__ vec2d operator*(double s) const {
		return vec2d(x * s, y * s);
	}
    __host__ __device__ vec2d& operator+=(const vec2d& b) {
        x += b.x;
        y += b.y;
        return *this;
    }
	__host__ __device__ double dot(const vec2d& b) const {
		return x * b.x + y * b.y;
	}
	__host__ __device__ double norm() const {
		return sqrt(x * x + y * y);
	}
    __host__ __device__ vec2d normalized() const {
        double len = norm();
        return (len > 1e-12) ? (*this * (1.0/len)) : vec2d(0, 0);
    }
};

struct vec3d {
	double x, y, z;

	__host__ __device__ vec3d() : x(0), y(0), z(0) {}
	__host__ __device__ vec3d(double x, double y, double z) : x(x), y(y), z(z) {}

	__host__ __device__ vec3d operator+(const vec3d& b) const {
		return vec3d(x + b.x, y + b.y, z + b.z);
	}
	__host__ __device__ vec3d operator-(const vec3d& b) const {
		return vec3d(x - b.x, y - b.y, z - b.z);
	}
	__host__ __device__ vec3d operator*(double s) const {
		return vec3d(x * s, y * s, z * s);
	}
    __host__ __device__ vec3d& operator+=(const vec3d& b) {
        x += b.x;
        y += b.y;
        z += b.z;
        return *this;
    }
	__host__ __device__ double dot(const vec3d& b) const {
		return x * b.x + y * b.y + z * b.z;
	}
	__host__ __device__ vec3d cross(const vec3d& b) const {
		return vec3d(
			y * b.z - z * b.y,
			z * b.x - x * b.z,
			x * b.y - y * b.x
		);
	}
	__host__ __device__ double norm() const {
		return sqrt(x * x + y * y + z * z);
	}
    __host__ __device__ vec3d normalized() const {
        double len = norm();
        return (len > 1e-12) ? (*this * (1.0/len)) : vec3d(0, 0, 0);
    }
};

void cpy(const double* A, double* B, const int N);