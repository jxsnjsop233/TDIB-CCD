#pragma once
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include <cuda_runtime.h>
#include "myEigen.cuh"
#include "paramBound.cuh"

struct CudaRecCubicBezier{
    static const int order = 3;
	static const int cntCp = 16;
    // control point order: 
    // 00, 01, 02, 03, 
    // 10, 11, 12, 13, 
    // 20, 21, 22, 23, 
    // 30, 31, 32, 33
	vec3d ctrlp[cntCp]; 

    __host__ __device__
    vec3d lerp(double t, const vec3d& t0, const vec3d& t1) const {
        return t0 * (1 - t) + t1 * t;
    }

    __host__ __device__
    vec3d blossomCubicBezier(const vec3d* p, double u0, double u1, double u2) const {
        vec3d a[3] = { lerp(u0, p[0], p[1]), lerp(u0, p[1], p[2]), lerp(u0, p[2], p[3]) };
        vec3d b[2] = { lerp(u1, a[0], a[1]), lerp(u1, a[1], a[2]) };
        return lerp(u2, b[0], b[1]);
    }

    __host__ __device__
	vec3d blossomBicubicBezier(const vec3d* cp, vec2d uv0, vec2d uv1, vec2d uv2) const {
        vec3d q[4];
        for (int i = 0; i < 4; i++) {
            q[i] = blossomCubicBezier(cp + (i * 4), uv0.y, uv1.y, uv2.y);
        }
        return blossomCubicBezier(q, uv0.x, uv1.x, uv2.x);
    }

    __host__ __device__
    vec3d evaluatePatchPoint(const vec2d &uv) const {
        return blossomBicubicBezier(ctrlp, uv, uv, uv);
    }

    __host__ __device__
    static vec3d axisU(const vec3d* pt) {
        return pt[12]-pt[0]+pt[15]-pt[3];
    }

    __host__ __device__
    static vec3d axisV(const vec3d* pt){
		return pt[3]-pt[0]+pt[15]-pt[12];
	}

    __host__ __device__
    void divideBezierPatch(CudaRecParamBound const uvB, vec3d* divCp) const {
        // vec3d divCp[16];
		divCp[0] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(0), uvB.corner(0));
		divCp[1] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(0), uvB.corner(2));
		divCp[2] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(2), uvB.corner(2));
		divCp[3] = blossomBicubicBezier(ctrlp, uvB.corner(2), uvB.corner(2), uvB.corner(2));
		divCp[4] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(0), uvB.corner(1));
		divCp[5] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(0), uvB.corner(3));
		divCp[6] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(2), uvB.corner(3));
		divCp[7] = blossomBicubicBezier(ctrlp, uvB.corner(2), uvB.corner(2), uvB.corner(3));
		divCp[8] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(1), uvB.corner(1));
		divCp[9] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(1), uvB.corner(3));
		divCp[10] = blossomBicubicBezier(ctrlp, uvB.corner(0), uvB.corner(3), uvB.corner(3));
		divCp[11] = blossomBicubicBezier(ctrlp, uvB.corner(2), uvB.corner(3), uvB.corner(3));
		divCp[12] = blossomBicubicBezier(ctrlp, uvB.corner(1), uvB.corner(1), uvB.corner(1));
		divCp[13] = blossomBicubicBezier(ctrlp, uvB.corner(1), uvB.corner(1), uvB.corner(3));
		divCp[14] = blossomBicubicBezier(ctrlp, uvB.corner(1), uvB.corner(3), uvB.corner(3));
		divCp[15] = blossomBicubicBezier(ctrlp, uvB.corner(3), uvB.corner(3), uvB.corner(3));
		// return divCp;
    }
};