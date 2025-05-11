# pragma once
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include <cuda_runtime.h>
#include "myEigen.cuh"

struct CudaRecParamBound {
    vec2d pMin, pMax;

    __host__ __device__
    CudaRecParamBound(const vec2d p1 = vec2d(0,0), const vec2d p2 = vec2d(1,1)) :
        pMin {p1.x<p2.x?p1.x:p2.x, p1.y<p2.y?p1.y:p2.y},
        pMax {p1.x>p2.x?p1.x:p2.x, p1.y>p2.y?p1.y:p2.y} {
    }

    __host__ __device__
    vec2d operator[](int i) const { return i == 0 ? pMin : pMax; }

    __host__ __device__
    CudaRecParamBound operator&(CudaRecParamBound const o) const {
        CudaRecParamBound ret;
		ret.pMin = vec2d(pMin.x < o.pMin.x ? pMin.x : o.pMin.x, pMin.y < o.pMin.y ? pMin.y : o.pMin.y);
		ret.pMax = vec2d(pMax.x > o.pMax.x ? pMax.x : o.pMax.x, pMax.y > o.pMax.y ? pMax.y : o.pMax.y);
		return ret;
    }

    __host__ __device__
    bool isDegenerate() const { return pMin.x > pMax.x || pMin.y > pMax.y; }

    __host__ __device__
    bool isInside(vec2d const &o) const { return (o.x >= pMin.x && o.x <= pMax.x && o.y >= pMin.y && o.y <= pMax.y); }

    __host__ __device__
    vec2d diagonal() const { return pMax - pMin; }

    __host__ __device__
    vec2d corner(int i) const { return vec2d((*this)[(i & 1)].x, (*this)[(i & 2) ? 1 : 0].y); }

    __host__ __device__
	CudaRecParamBound interpSubpatchParam(const int id) const {
        // if (fabs(pMax.x - pMin.x) < 1e-8 && fabs(pMax.y - pMin.y) < 1e-8)
        //     return *this;  // 或者直接 skip
		vec2d pMid = (pMin + pMax) * 0.5;
		return CudaRecParamBound(corner(id), pMid);
	}

    __host__ __device__
    vec2d cudaCenterParam() const { return (pMin + pMax) * 0.5; }

    __host__ __device__
    vec2d centerParam() const {
        return vec2d(0.5 * (pMin.x + pMax.x), 0.5 * (pMin.y + pMax.y));
    }

    __host__ __device__
    double width() const {
		return ((pMax - pMin).x>(pMax - pMin).y)?(pMax - pMin).x:(pMax - pMin).y;
	}
};