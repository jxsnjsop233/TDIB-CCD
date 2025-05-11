#include <cuda_runtime.h>
#include "myEigen.cuh"
#include "recBezier.cuh"

#define KASE_NUM_MAX 100001

__global__ void arrayCpy(const double* A, double* B, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        B[idx] = A[idx];
    }
}

void cpy(const double* A, double* B, const int N) {
    vec3d v{1, 2, 0};
    double *d_A, *d_B;
    size_t size = N * sizeof(double);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    arrayCpy<<<blocks, threads>>>(d_A, d_B, N);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

__device__ bool primitiveCheckGPU ( const CudaRecCubicBezier CpPos1, const CudaRecCubicBezier CpVel1,
                                    const CudaRecCubicBezier CpPos2, const CudaRecCubicBezier CpVel2,
                                    const CudaRecParamBound divUvB1, const CudaRecParamBound divUvB2, 
                                    const int bb, 
                                    const vec2d divTime = vec2d(0, 1)) {
    vec3d posStart1[16], posEnd1[16], ptVel1[16], posStart2[16], posEnd2[16], ptVel2[16];
    CpPos1.divideBezierPatch(divUvB1, posStart1);
    CpPos1.divideBezierPatch(divUvB1, posEnd1);
    CpVel1.divideBezierPatch(divUvB1, ptVel1);
    CpPos2.divideBezierPatch(divUvB2, posStart2);
    CpPos2.divideBezierPatch(divUvB2, posEnd2);
    CpVel2.divideBezierPatch(divUvB2, ptVel2);

    // printf("111\n");

    for (int j = 0; j < CudaRecCubicBezier::cntCp; j++) {
        posStart1[j] += ptVel1[j] * divTime.x;
        posEnd1[j] += ptVel1[j] * divTime.y;
        posStart2[j] += ptVel2[j] * divTime.x;
        posEnd2[j] += ptVel2[j] * divTime.y;
    }

    vec3d axes[15];
    int axesNum = 15;
    if(bb==0){
        axesNum = 3;
		axes[0] = vec3d(1,0,0); axes[1] = vec3d(0,1,0); axes[2] = vec3d(0,0,1);
	}
    else if(bb==1){
        vec3d lu1 = CudaRecCubicBezier::axisU(posStart1);
        vec3d lv1tmp = CudaRecCubicBezier::axisV(posStart1);
        vec3d ln1 = lu1.cross(lv1tmp);
        vec3d lv1 = ln1.cross(lu1);

        vec3d lu2 = CudaRecCubicBezier::axisU(posStart2);
        vec3d lv2tmp = CudaRecCubicBezier::axisV(posStart2);
        vec3d ln2 = lu2.cross(lv2tmp);
        vec3d lv2 = ln2.cross(lu2);
        
        // SAT
        axes[0] = lu1; axes[1] = lv1; axes[2] = ln1;
        axes[3] = lu2; axes[4] = lv2; axes[5] = ln2;
        axes[6] = lu1.cross(lu2); axes[7] = lu1.cross(lv2); axes[8] = lu1.cross(ln2);
        axes[9] = lv1.cross(lu2); axes[10] = lv1.cross(lv2); axes[11] = lv1.cross(ln2);
        axes[12] = ln1.cross(lu2); axes[13] = ln1.cross(lv2); axes[14] = ln1.cross(ln2);
    }

    for(int i=0; i<axesNum; i++){
        double maxProj1 = -99999999.9, minProj1 = 99999999.9;
        for(int j = 0; j < CudaRecCubicBezier::cntCp; j++){
            maxProj1 = max(maxProj1, posStart1[j].dot(axes[i]));
            minProj1 = min(minProj1, posStart1[j].dot(axes[i]));
        }
        for(int j = 0; j < CudaRecCubicBezier::cntCp; j++){
            maxProj1 = max(maxProj1, posEnd1[j].dot(axes[i]));
            minProj1 = min(minProj1, posEnd1[j].dot(axes[i]));
        }
        double maxProj2 = -99999999.9, minProj2 = 99999999.9;
        for(int j = 0; j < CudaRecCubicBezier::cntCp; j++){
            maxProj2 = max(maxProj2, posStart2[j].dot(axes[i]));
            minProj2 = min(minProj2, posStart2[j].dot(axes[i]));
        }
        for(int j = 0; j < CudaRecCubicBezier::cntCp; j++){
            maxProj2 = max(maxProj2, posEnd2[j].dot(axes[i]));
            minProj2 = min(minProj2, posEnd2[j].dot(axes[i]));
        }
        if(maxProj2<minProj1 || maxProj1<minProj2) return false;
    }
    return true;
}

__global__ void solveCCDKernel ( 
    const CudaRecCubicBezier *CpPos1List, const CudaRecCubicBezier *CpVel1List,
    const CudaRecCubicBezier *CpPos2List, const CudaRecCubicBezier *CpVel2List,
    vec2d *uv1Out, vec2d *uv2Out,
    double* timeOut, const int bb,
    const double deltaDist = 1e-5,
    const double upperTime = 1, const int nPairs = 1) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nPairs) return;
    
    const CudaRecCubicBezier CpPos1 = CpPos1List[idx];
    const CudaRecCubicBezier CpVel1 = CpVel1List[idx];
    const CudaRecCubicBezier CpPos2 = CpPos2List[idx];
    const CudaRecCubicBezier CpVel2 = CpVel2List[idx];

    CudaRecParamBound initPB1, initPB2;
    vec2d initTimeIntv(0, upperTime);
    if (!primitiveCheckGPU(CpPos1, CpVel1, CpPos2, CpVel2, initPB1, initPB2, bb, initTimeIntv)) {
        timeOut[idx] = -1.0;
        return;
    }

    // 简化的固定深度 DFS 栈
    const int maxStackSize = 64;
    CudaRecParamBound pb1Stack[maxStackSize];
    CudaRecParamBound pb2Stack[maxStackSize];
    vec2d tStack[maxStackSize];
    int stackTop = 0;

    pb1Stack[stackTop] = initPB1;
    pb2Stack[stackTop] = initPB2;
    tStack[stackTop] = initTimeIntv;
    stackTop++;

    while (stackTop > 0) {
        // printf("%d ",stackTop);
        stackTop--;
        CudaRecParamBound curPB1 = pb1Stack[stackTop];
        CudaRecParamBound curPB2 = pb2Stack[stackTop];
        vec2d curT = tStack[stackTop];

        double width = max(max(curPB1.width(), curPB2.width()), curT.y - curT.x);
        
        if (width < deltaDist) {
            uv1Out[idx] = curPB1.cudaCenterParam();
            uv2Out[idx] = curPB2.cudaCenterParam();
            timeOut[idx] = curT.x;
            return;
        }

        double tMid = 0.5 * (curT.x + curT.y);
        vec2d divTime1(curT.x, tMid), divTime2(tMid, curT.y);

        for (int i = 0; i < 4 && stackTop + 16 < maxStackSize; ++i) {
            CudaRecParamBound subPB1 = curPB1.interpSubpatchParam(i);
            for (int j = 0; j < 4; ++j) {
                CudaRecParamBound subPB2 = curPB2.interpSubpatchParam(j);
                if (primitiveCheckGPU(CpPos1, CpVel1, CpPos2, CpVel2, subPB1, subPB2, bb, divTime1)) {
                    pb1Stack[stackTop] = subPB1;
                    pb2Stack[stackTop] = subPB2;
                    tStack[stackTop] = divTime1;
                    // printf("++\n");
                    stackTop++;
                }
                if (primitiveCheckGPU(CpPos1, CpVel1, CpPos2, CpVel2, subPB1, subPB2, bb, divTime2)) {
                    pb1Stack[stackTop] = subPB1;
                    pb2Stack[stackTop] = subPB2;
                    tStack[stackTop] = divTime2;
                    // printf("++\n");
                    stackTop++;
                }
            }
        }
    }

    timeOut[idx] = -1.0;
    // return timeOut[idx];
}

void cudaSolveCCD ( const CudaRecCubicBezier *CpPos1, const CudaRecCubicBezier *CpVel1,
                    const CudaRecCubicBezier *CpPos2, const CudaRecCubicBezier *CpVel2,
                    vec2d *uv1, vec2d *uv2,
                    const int bb, const int kaseNum,
                    const double deltaDist = 1e-5,
                    const double upperTime = 1, int maxRounds = 10){
    int nPairs = kaseNum;
    double timeOut[KASE_NUM_MAX];
    

    // 分配 device memory
    CudaRecCubicBezier *d_CpPos1, *d_CpVel1, *d_CpPos2, *d_CpVel2;
    vec2d *d_uv1, *d_uv2;
    double *d_time;
    
    cudaError_t errAsync = cudaGetLastError();
    // printf("----0----\n");
    // fprintf(stderr, "CUDA async kernel launch error: %s\n", cudaGetErrorString(errAsync));

    cudaMalloc(&d_CpPos1, sizeof(CudaRecCubicBezier) * nPairs);
    cudaMalloc(&d_CpVel1, sizeof(CudaRecCubicBezier) * nPairs);
    cudaMalloc(&d_CpPos2, sizeof(CudaRecCubicBezier) * nPairs);
    cudaMalloc(&d_CpVel2, sizeof(CudaRecCubicBezier) * nPairs);
    
    errAsync = cudaGetLastError();
    // printf("----1----\n");
    // fprintf(stderr, "CUDA async kernel launch error: %s\n", cudaGetErrorString(errAsync));

    cudaMalloc(&d_uv1, sizeof(vec2d) * nPairs);
    cudaMalloc(&d_uv2, sizeof(vec2d) * nPairs);
    cudaMalloc(&d_time, sizeof(double) * nPairs);

    // 拷贝数据到 GPU
    cudaMemcpy(d_CpPos1, CpPos1, sizeof(CudaRecCubicBezier) * nPairs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CpVel1, CpVel1, sizeof(CudaRecCubicBezier) * nPairs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CpPos2, CpPos2, sizeof(CudaRecCubicBezier) * nPairs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CpVel2, CpVel2, sizeof(CudaRecCubicBezier) * nPairs, cudaMemcpyHostToDevice);
    // printf("----2----\n");
    errAsync = cudaGetLastError();
    // fprintf(stderr, "CUDA async kernel launch error: %s\n", cudaGetErrorString(errAsync));
    solveCCDKernel<<<(nPairs + 255)/256, 256>>>(
        d_CpPos1, d_CpVel1, d_CpPos2, d_CpVel2,
        d_uv1, d_uv2, d_time, bb, deltaDist, upperTime, nPairs
    );
    // 同步 CUDA，等待 kernel 执行完成
    cudaError_t errSync  = cudaDeviceSynchronize();
    errAsync = cudaGetLastError();
    // printf("----3----\n");
    // fprintf(stderr, "CUDA async kernel launch error: %s\n", cudaGetErrorString(errAsync));

    if (errSync != cudaSuccess) {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess) {
        fprintf(stderr, "CUDA async kernel launch error: %s\n", cudaGetErrorString(errAsync));
    }

    // 取回结果
    cudaMemcpy(uv1, d_uv1, sizeof(vec2d) * nPairs, cudaMemcpyDeviceToHost);
    cudaMemcpy(uv2, d_uv2, sizeof(vec2d) * nPairs, cudaMemcpyDeviceToHost);
    cudaMemcpy(timeOut, d_time, sizeof(double) * nPairs, cudaMemcpyDeviceToHost); 


    // 清理资源
    cudaFree(d_CpPos1);
    cudaFree(d_CpVel1);
    cudaFree(d_CpPos2);
    cudaFree(d_CpVel2);
    cudaFree(d_uv1);
    cudaFree(d_uv2);
    cudaFree(d_time);

    int successCount = 0;
    for (int i = 0; i < nPairs; ++i){
        if (timeOut[i] >= 0) successCount++;
        // printf("timeOut[%d]: %d.\n", i, timeOut[i]);
    }
    printf("success cases num is: %d.\n", successCount);
}
