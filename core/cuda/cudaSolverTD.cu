#include <cuda_runtime.h>
#include "myEigen.cuh"
#include "recBezier.cuh"

#define MAX_CP 16
#define MAX_LINES 32
#define MAX_FEASIBLE 32
#define KASE_NUM_MAX 100001

struct Line {
    double k, b;
    __device__ Line() : k(0), b(0) {}
    __device__ Line(double _k, double _b) : k(_k), b(_b) {}
};

__device__ void reverseLines(Line *arr, int count) {
    for (int i = 0; i < count / 2; ++i) {
        Line tmp = arr[i];
        arr[i] = arr[count - 1 - i];
        arr[count - 1 - i] = tmp;
    }
}

__device__ void sortLines(Line *arr, int count) {
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (arr[j].k < arr[i].k || (arr[j].k == arr[i].k && arr[j].b > arr[i].b)) {
                Line tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
        }
    }
}

__device__ int calcBoundaries_GPU(Line *lines, int count, Line *ch, bool getMaxCH, vec2d tIntv) {
    if (!getMaxCH) reverseLines(lines, count);

    int chCount = 0;
    ch[chCount++] = lines[0];
    int alpha = 1;
    while (alpha < count) {
        int beta = chCount - 1;
        while (beta > 0) {
            double chfp = (ch[beta].k - ch[beta - 1].k) * (lines[alpha].b - ch[beta - 1].b)
                        - (lines[alpha].k - ch[beta - 1].k) * (ch[beta].b - ch[beta - 1].b);
            if (chfp >= 0) {
                --chCount;
                --beta;
            } else break;
        }
        if (beta == 0) {
            double chStart = tIntv.x * (lines[alpha].k - ch[0].k) + (lines[alpha].b - ch[0].b);
            if ((getMaxCH && chStart >= 0) || (!getMaxCH && chStart <= 0)) chCount = 0;
        }
        ch[chCount++] = lines[alpha];
        ++alpha;
    }
    return chCount;
}

__device__ vec2d boundaryIntersect_GPU(const Line* ch1, int ch1Count, const Line* ch2, int ch2Count, vec2d tIntv) {
    int id1 = 0, id2 = 0;
    double intvL = -1.0, intvR = -1.0;

    // ======== 寻找交集左端点 ========
    if (ch1[0].k * tIntv.x + ch1[0].b < ch2[0].k * tIntv.x + ch2[0].b) {
        intvL = tIntv.x;
    } else {
        while (id1 < ch1Count && id2 < ch2Count) {
            if (ch1[id1].k >= ch2[id2].k) break;

            double hifp1, hifp2;
            if (id1 < ch1Count - 1)
                hifp1 = (ch1[id1 + 1].k - ch2[id2].k) * (ch1[id1].b - ch2[id2].b)
                      - (ch1[id1].k - ch2[id2].k) * (ch1[id1 + 1].b - ch2[id2].b);
            else
                hifp1 = tIntv.y * (ch1[id1].k - ch2[id2].k) + (ch1[id1].b - ch2[id2].b);

            if (id2 < ch2Count - 1)
                hifp2 = (ch1[id1].k - ch2[id2 + 1].k) * (ch1[id1].b - ch2[id2].b)
                      - (ch1[id1].k - ch2[id2].k) * (ch1[id1].b - ch2[id2 + 1].b);
            else
                hifp2 = tIntv.y * (ch1[id1].k - ch2[id2].k) + (ch1[id1].b - ch2[id2].b);

            if (hifp1 < 0) {
                if (hifp2 < 0) {
                    double denom = ch1[id1].k - ch2[id2].k;
                    if (fabs(denom) < 1e-12) return vec2d(-1.0, -1.0);
                    intvL = -(ch1[id1].b - ch2[id2].b) / denom;
                    break;
                } else {
                    id2++;
                }
            } else {
                id1++;
                if (!(hifp2 < 0)) id2++;
            }
        }

        if (intvL == -1.0 || intvL >= tIntv.y)
            return vec2d(-1.0, -1.0);
    }

    // ======== 寻找交集右端点 ========
    id1 = ch1Count - 1;
    id2 = ch2Count - 1;

    if ((ch1[id1].k - ch2[id2].k) * tIntv.y + (ch1[id1].b - ch2[id2].b) < 0) {
        intvR = tIntv.y;
    } else {
        while (id1 >= 0 && id2 >= 0) {
            if (ch1[id1].k <= ch2[id2].k) return vec2d(-1.0, -1.0);

            double hifp1, hifp2;
            if (id1 > 0)
                hifp1 = (ch1[id1].k - ch2[id2].k) * (ch1[id1 - 1].b - ch2[id2].b)
                      - (ch1[id1 - 1].k - ch2[id2].k) * (ch1[id1].b - ch2[id2].b);
            else
                hifp1 = tIntv.x * (ch1[id1].k - ch2[id2].k) + (ch1[id1].b - ch2[id2].b);

            if (id2 > 0)
                hifp2 = (ch1[id1].k - ch2[id2].k) * (ch1[id1].b - ch2[id2 - 1].b)
                      - (ch1[id1].k - ch2[id2 - 1].k) * (ch1[id1].b - ch2[id2].b);
            else
                hifp2 = tIntv.x * (ch1[id1].k - ch2[id2].k) + (ch1[id1].b - ch2[id2].b);

            if (hifp1 < 0) {
                if (hifp2 < 0) {
                    double denom = ch1[id1].k - ch2[id2].k;
                    if (fabs(denom) < 1e-12) return vec2d(-1.0, -1.0);
                    intvR = -(ch1[id1].b - ch2[id2].b) / denom;
                    break;
                } else {
                    id2--;
                }
            } else {
                id1--;
                if (!(hifp2 < 0)) id2--;
            }
        }

        if (intvR == -1.0 || intvR <= intvL) return vec2d(-1.0, -1.0);
    }

    // ======== 最终边界判断 ========
    if (intvL > intvR || intvL < tIntv.x || intvR > tIntv.y) {
        return vec2d(-1.0, -1.0);
    }

    return vec2d(intvL, intvR);
}

__device__ bool primitiveCheckTD_GPU(const CudaRecCubicBezier CpPos1, const CudaRecCubicBezier CpVel1,
                                     const CudaRecCubicBezier CpPos2, const CudaRecCubicBezier CpVel2,
                                     const CudaRecParamBound divUvB1, const CudaRecParamBound divUvB2,
                                     vec2d &colTime, const int bb, const vec2d &initTimeIntv) {
    vec3d ptPos1[MAX_CP], ptVel1[MAX_CP], ptPos2[MAX_CP], ptVel2[MAX_CP];
    CpPos1.divideBezierPatch(divUvB1, ptPos1);
    CpVel1.divideBezierPatch(divUvB1, ptVel1);
    CpPos2.divideBezierPatch(divUvB2, ptPos2);
    CpVel2.divideBezierPatch(divUvB2, ptVel2);

    
    // printf("111111\n");

    vec2d timeIntv(initTimeIntv.x - 1e-6, initTimeIntv.y + 1e-6);

    vec3d axes[15];
    int axesNum = 0;
    if (bb == 0) {
        axes[0] = vec3d(1, 0, 0); axes[1] = vec3d(0, 1, 0); axes[2] = vec3d(0, 0, 1);
        axesNum = 3;
    } else if(bb==1){
         double t = initTimeIntv.x;

        // 计算 patch1 的主方向
        vec3d lu1 = CudaRecCubicBezier::axisU(ptPos1) + CudaRecCubicBezier::axisU(ptVel1) * t;
        vec3d lv1tmp = CudaRecCubicBezier::axisV(ptPos1) + CudaRecCubicBezier::axisV(ptVel1) * t;
        vec3d ln1 = lu1.cross(lv1tmp);
        if (lu1.norm() < 1e-8 || lv1tmp.norm() < 1e-8 || ln1.norm() < 1e-8) return false;  // Patch1退化
        vec3d lv1 = ln1.cross(lu1);
        if (lv1.norm() < 1e-8) return false;

        // 计算 patch2 的主方向
        vec3d lu2 = CudaRecCubicBezier::axisU(ptPos2) + CudaRecCubicBezier::axisU(ptVel2) * t;
        vec3d lv2tmp = CudaRecCubicBezier::axisV(ptPos2) + CudaRecCubicBezier::axisV(ptVel2) * t;
        vec3d ln2 = lu2.cross(lv2tmp);
        if (lu2.norm() < 1e-8 || lv2tmp.norm() < 1e-8 || ln2.norm() < 1e-8) return false;  // Patch2退化
        vec3d lv2 = ln2.cross(lu2);
        if (lv2.norm() < 1e-8) return false;

        // 归一化（可选）
        lu1 = lu1.normalized(); lv1 = lv1.normalized(); ln1 = ln1.normalized();
        lu2 = lu2.normalized(); lv2 = lv2.normalized(); ln2 = ln2.normalized();

        // 构造 15 个投影轴
        axes[0] = lu1; axes[1] = lv1; axes[2] = ln1;
        axes[3] = lu2; axes[4] = lv2; axes[5] = ln2;
        axes[6] = lu1.cross(lu2); axes[7] = lu1.cross(lv2); axes[8] = lu1.cross(ln2);
        axes[9] = lv1.cross(lu2); axes[10] = lv1.cross(lv2); axes[11] = lv1.cross(ln2);
        axes[12] = ln1.cross(lu2); axes[13] = ln1.cross(lv2); axes[14] = ln1.cross(ln2);

        // 过滤零向量
        axesNum = 0;
        for (int i = 0; i < 15; ++i) {
            if (axes[i].norm() > 1e-8) {
                axes[axesNum++] = axes[i].normalized();
            }
        }

        if (axesNum == 0) return false;  // 所有投影轴退化
    }

    vec2d feasible[MAX_FEASIBLE];
    int feasibleCount = 0;

    for (int a = 0; a < axesNum; ++a) {
        Line ptLines1[MAX_LINES], ptLines2[MAX_LINES];
        int n1 = CudaRecCubicBezier::cntCp;
        for (int i = 0; i < n1; ++i)
            ptLines1[i] = Line(ptVel1[i].dot(axes[a]), ptPos1[i].dot(axes[a]) + 1e-12 * fabs(ptPos1[i].dot(axes[a])));
        for (int i = 0; i < n1; ++i)
            ptLines2[i] = Line(ptVel2[i].dot(axes[a]), ptPos2[i].dot(axes[a]) - 1e-12 * fabs(ptPos2[i].dot(axes[a])));

        sortLines(ptLines1, n1);
        sortLines(ptLines2, n1);

        Line ch1[MAX_LINES], ch2[MAX_LINES];
        int c1 = calcBoundaries_GPU(ptLines1, n1, ch1, true, timeIntv);
        int c2 = calcBoundaries_GPU(ptLines2, n1, ch2, false, timeIntv);
        if (c1 == 0 || c2 == 0) continue; // 空壳
        vec2d intvT1 = boundaryIntersect_GPU(ch1, c1, ch2, c2, timeIntv);

        Line ch3[MAX_LINES], ch4[MAX_LINES];
        int c3 = calcBoundaries_GPU(ptLines2, n1, ch3, true, timeIntv);
        int c4 = calcBoundaries_GPU(ptLines1, n1, ch4, false, timeIntv);
        if (c3 == 0 || c4 == 0) continue; // 空壳
        vec2d intvT2 = boundaryIntersect_GPU(ch3, c3, ch4, c4, timeIntv);
        
        if(intvT1.x!=-1) feasible[feasibleCount++] = intvT1; 
        if(intvT2.x!=-1) feasible[feasibleCount++] = intvT2; 
    }

    // printf("%d\n",feasibleCount);
    if (feasibleCount == 0) {
        colTime = initTimeIntv;
        return true;
    }

    double minT = initTimeIntv.x, maxT = initTimeIntv.y;
    // Find the lower bound
    // Sort by feasible[i].x ascending
    for (int i = 0; i < feasibleCount - 1; ++i) {
        for (int j = i + 1; j < feasibleCount; ++j) {
            if (feasible[i].x > feasible[j].x) {
                vec2d temp = feasible[i];
                feasible[i] = feasible[j];
                feasible[j] = temp;
            }
        }
    }
    if (feasible[0].x < initTimeIntv.x) {
        minT = fmax(minT, feasible[0].y);
        for (int i = 1; i < feasibleCount; ++i) {
            if (feasible[i].x < minT)
                minT = fmax(minT, feasible[i].y);
            else break;
        }
    }
    if (minT > maxT) {colTime = vec2d(-1, -1); return false;}

    // Find the upper bound
    // Sort by feasible[i].y descending
    for (int i = 0; i < feasibleCount - 1; ++i) {
        for (int j = i + 1; j < feasibleCount; ++j) {
            if (feasible[i].y < feasible[j].y) {
                vec2d temp = feasible[i];
                feasible[i] = feasible[j];
                feasible[j] = temp;
            }
        }
    }
    if (feasible[0].y > initTimeIntv.y) {
        maxT = fmin(maxT, feasible[0].x);
        for (int i = 1; i < feasibleCount; ++i) {
            if (feasible[i].y > maxT)
                maxT = fmin(maxT, feasible[i].x);
            else break;
        }
    }

    if (minT >= maxT)
        colTime = vec2d(maxT, minT);
    else
        colTime = vec2d(minT, maxT);
    return true;

}

__global__ void solveCCDTDKernel(
    const CudaRecCubicBezier *CpPos1List, const CudaRecCubicBezier *CpVel1List,
    const CudaRecCubicBezier *CpPos2List, const CudaRecCubicBezier *CpVel2List,
    vec2d *uv1Out, vec2d *uv2Out,
    double* timeOut, const int bb,
    const double deltaDist = 1e-5,
    const double upperTime = 1, const int nPairs = 1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("----00----\n");
    if (idx >= nPairs) return;
    // printf("----01----\n");

    const CudaRecCubicBezier CpPos1 = CpPos1List[idx];
    const CudaRecCubicBezier CpVel1 = CpVel1List[idx];
    const CudaRecCubicBezier CpPos2 = CpPos2List[idx];
    const CudaRecCubicBezier CpVel2 = CpVel2List[idx];

    CudaRecParamBound initPB1, initPB2;
    vec2d initTimeIntv(0, upperTime), colTime;

    if (!primitiveCheckTD_GPU(CpPos1, CpVel1, CpPos2, CpVel2, initPB1, initPB2, colTime, bb, initTimeIntv)) {
        timeOut[idx] = -1.0;
        return;
    }
    
    // printf("----02----\n");

    const int maxStackSize = 128;
    CudaRecParamBound pb1Stack[maxStackSize];
    CudaRecParamBound pb2Stack[maxStackSize];
    vec2d tStack[maxStackSize];
    int stackTop = 0;

    pb1Stack[stackTop] = initPB1;
    pb2Stack[stackTop] = initPB2;
    tStack[stackTop] = colTime;
    stackTop++;

    // printf("----03----\n");

    while (stackTop > 0) {
        stackTop--;
        CudaRecParamBound curPB1 = pb1Stack[stackTop];
        CudaRecParamBound curPB2 = pb2Stack[stackTop];
        vec2d curT = tStack[stackTop];

        double width = max(max(curPB1.width(), curPB2.width()), curT.y - curT.x);
        // printf("%lf\n",width);
        if (width < deltaDist) {
            uv1Out[idx] = curPB1.cudaCenterParam();
            uv2Out[idx] = curPB2.cudaCenterParam();
            timeOut[idx] = curT.x;
            return;
        }

        // if(stackTop<40){
        //     printf("%d %lf %lf %lf %lf \n", stackTop, curPB1.width(), curPB2.width(), curT.x, curT.y);
        // }
        // printf("%lf %lf %lf %lf \n", curPB1.width(), curPB2.width(), curT.x, curT.y);
        // printf("%lf %lf %lf %lf \n", curPB1.pMin, curPB1.pMax, curPB2.pMin, curPB2.pMax);

        for (int i = 0; i < 4 && stackTop + 4 * 4 < maxStackSize; ++i) {
            CudaRecParamBound subPB1 = curPB1.interpSubpatchParam(i);
            for (int j = 0; j < 4; ++j) {
                CudaRecParamBound subPB2 = curPB2.interpSubpatchParam(j);
                vec2d newColTime;
                if (primitiveCheckTD_GPU(CpPos1, CpVel1, CpPos2, CpVel2, subPB1, subPB2, newColTime, bb, curT)) {
                    pb1Stack[stackTop] = subPB1;
                    pb2Stack[stackTop] = subPB2;
                    tStack[stackTop] = newColTime;
                    // printf("%lf %lf \n", newColTime.x, newColTime.y);
                    stackTop++;
                }
            }
        }
    }

    timeOut[idx] = -1.0;
}

void cudaSolveCCD_TD(const CudaRecCubicBezier *CpPos1, const CudaRecCubicBezier *CpVel1,
                     const CudaRecCubicBezier *CpPos2, const CudaRecCubicBezier *CpVel2,
                     vec2d *uv1, vec2d *uv2,
                     const int bb, const int kaseNum,
                     const double deltaDist = 1e-5,
                     const double upperTime = 1.0) {
    int nPairs = kaseNum;
    double timeOut[KASE_NUM_MAX];  // assume kaseNum <= 10001

    // Allocate device memory
    CudaRecCubicBezier *d_CpPos1, *d_CpVel1, *d_CpPos2, *d_CpVel2;
    vec2d *d_uv1, *d_uv2;
    double *d_time;

    cudaMalloc(&d_CpPos1, sizeof(CudaRecCubicBezier) * nPairs);
    cudaMalloc(&d_CpVel1, sizeof(CudaRecCubicBezier) * nPairs);
    cudaMalloc(&d_CpPos2, sizeof(CudaRecCubicBezier) * nPairs);
    cudaMalloc(&d_CpVel2, sizeof(CudaRecCubicBezier) * nPairs);
    cudaMalloc(&d_uv1, sizeof(vec2d) * nPairs);
    cudaMalloc(&d_uv2, sizeof(vec2d) * nPairs);
    cudaMalloc(&d_time, sizeof(double) * nPairs);

    // Copy input data to device
    cudaMemcpy(d_CpPos1, CpPos1, sizeof(CudaRecCubicBezier) * nPairs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CpVel1, CpVel1, sizeof(CudaRecCubicBezier) * nPairs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CpPos2, CpPos2, sizeof(CudaRecCubicBezier) * nPairs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_CpVel2, CpVel2, sizeof(CudaRecCubicBezier) * nPairs, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(256);
    dim3 grid((nPairs + block.x - 1) / block.x);
    // printf("----1----\n");
    solveCCDTDKernel<<<grid, block>>>(
        d_CpPos1, d_CpVel1, d_CpPos2, d_CpVel2,
        d_uv1, d_uv2, d_time, bb, deltaDist, upperTime, nPairs
    );
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(uv1, d_uv1, sizeof(vec2d) * nPairs, cudaMemcpyDeviceToHost);
    cudaMemcpy(uv2, d_uv2, sizeof(vec2d) * nPairs, cudaMemcpyDeviceToHost);
    cudaMemcpy(timeOut, d_time, sizeof(double) * nPairs, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_CpPos1); cudaFree(d_CpVel1);
    cudaFree(d_CpPos2); cudaFree(d_CpVel2);
    cudaFree(d_uv1);    cudaFree(d_uv2);
    cudaFree(d_time);

    // Count success cases (optional debug info)
    int successCount = 0;
    for (int i = 0; i < nPairs; ++i) {
        if (timeOut[i] >= 0) successCount++;
    }
    printf("[TD-GPU] success cases: %d / %d\n", successCount, nPairs);
}