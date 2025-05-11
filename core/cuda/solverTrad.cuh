#pragma once
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include <cuda_runtime.h>
#include "recBezier.cuh"

__device__ double max(double a, double b) {return a>b?a:b;}
__device__ double min(double a, double b) {return a<b?a:b;}

__device__ bool primitiveCheckGPU ( const CudaRecCubicBezier CpPos1, const CudaRecCubicBezier CpVel1,
                                    const CudaRecCubicBezier CpPos2, const CudaRecCubicBezier CpVel2,
                                    const CudaRecParamBound divUvB1, const CudaRecParamBound divUvB2, 
                                    const int bb, 
                                    const vec2d divTime = vec2d(0, 1)) ;

__global__ void solveCCDKernel ( const CudaRecCubicBezier *CpPos1List, const CudaRecCubicBezier *CpVel1List,
                    const CudaRecCubicBezier *CpPos2List, const CudaRecCubicBezier *CpVel2List,
                    vec2d *uv1Out, vec2d *uv2Out,
                    double* timeOut, const int bb,
                    const double deltaDist = 1e-5,
                    const double upperTime = 1, const int nPairs = 1) ;

void cudaSolveCCD ( const CudaRecCubicBezier *CpPos1, const CudaRecCubicBezier *CpVel1,
                    const CudaRecCubicBezier *CpPos2, const CudaRecCubicBezier *CpVel2,
                    vec2d *uv1, vec2d *uv2,
                    const int bb, const int kaseNum,
                    const double deltaDist,
                    const double upperTime = 1, int maxRounds = 10) ;