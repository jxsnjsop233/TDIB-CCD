#pragma once
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include <cuda_runtime.h>
#include "myEigen.cuh"
#include "paramBound.cuh"

// __device__ void reverseLines(Line *arr, int count);

// __device__ void sortLines(Line *arr, int count);

// __device__ int calcBoundaries_GPU(Line *lines, int count, Line *ch, bool getMaxCH, vec2d tIntv);

// __device__ vec2d boundaryIntersect_GPU(const Line* ch1, int ch1Count, const Line* ch2, int ch2Count, vec2d tIntv);

// __device__ bool primitiveCheckTD_GPU(const CudaRecCubicBezier CpPos1, const CudaRecCubicBezier CpVel1,
//                                      const CudaRecCubicBezier CpPos2, const CudaRecCubicBezier CpVel2,
//                                      const CudaRecParamBound divUvB1, const CudaRecParamBound divUvB2,
//                                      vec2d &colTime, const int bb, const vec2d &initTimeIntv);

// __global__ void solveCCDTDKernel(
//     const CudaRecCubicBezier *CpPos1List, const CudaRecCubicBezier *CpVel1List,
//     const CudaRecCubicBezier *CpPos2List, const CudaRecCubicBezier *CpVel2List,
//     vec2d *uv1Out, vec2d *uv2Out,
//     double* timeOut, const int bb,
//     const double deltaDist = 1e-5,
//     const double upperTime = 1, const int nPairs = 1);

void cudaSolveCCD_TD(const CudaRecCubicBezier *CpPos1, const CudaRecCubicBezier *CpVel1,
                     const CudaRecCubicBezier *CpPos2, const CudaRecCubicBezier *CpVel2,
                     vec2d *uv1, vec2d *uv2,
                     const int bb, const int kaseNum,
                     const double deltaDist = 1e-5,
                     const double upperTime = 1.0);
