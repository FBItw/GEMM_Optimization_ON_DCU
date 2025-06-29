#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <omp.h>
#include <math.h>

#define N 1024
#define TILE 16
#define RUNS 20

__global__ void myGEMM_optimized(float *A, float *B, float *C, int width) {
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < width / TILE; ++t) {
        tile_A[threadIdx.y][threadIdx.x] = A[row * width + t * TILE + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * width + col];
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        __syncthreads();
    }

    C[row * width + col] = sum;
}

void checkResult(float *hostRef, float *gpuRef, int n) {
    float epsilon = 1e-3f;
    float max_rel_error = 0.0f, sum_rel_error = 0.0f;
    bool flag = true;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(hostRef[i] - gpuRef[i]);
        float base = fabsf(hostRef[i]) > 1e-6f ? fabsf(hostRef[i]) : 1.0f;
        float rel_err = diff / base;
        if (rel_err > epsilon) flag = false;
        max_rel_error = fmaxf(max_rel_error, rel_err);
        sum_rel_error += rel_err;
    }
    if (flag)
        printf("计算正确 ✅\n");
    else {
        printf("计算错误 ❌\n");
        printf("最大相对误差: %e\n", max_rel_error);
        printf("平均相对误差: %e\n", sum_rel_error / n);
    }
}

int main() {
    int size = N * N;
    size_t bytes = size * sizeof(float);

    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C_naive = (float*)malloc(bytes);
    float *C_my = (float*)malloc(bytes);

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);

    srand(12345);
    for (int i = 0; i < size; i++) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
        C_naive[i] = 0.0f;
        C_my[i] = 0.0f;
    }

    hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // === DCU 平均时间 ===
    double total_dcu_time = 0.0;
    for (int i = 0; i < RUNS; ++i) {
        struct timeval mystart, myend;
        gettimeofday(&mystart, NULL);
        hipLaunchKernelGGL(myGEMM_optimized, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, N);
        hipDeviceSynchronize();
        gettimeofday(&myend, NULL);
        total_dcu_time += (myend.tv_sec - mystart.tv_sec) + (myend.tv_usec - mystart.tv_usec) * 1e-6;
    }
    hipMemcpy(C_my, d_C, bytes, hipMemcpyDeviceToHost);
    double avg_dcu_time = total_dcu_time / RUNS;

    // === CPU 平均时间 ===
    double total_cpu_time = 0.0;
    for (int r = 0; r < RUNS; ++r) {
        struct timeval ref_start, ref_end;
        gettimeofday(&ref_start, NULL);

#pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C_naive[i * N + j] = sum;
            }

        gettimeofday(&ref_end, NULL);
        total_cpu_time += (ref_end.tv_sec - ref_start.tv_sec) + (ref_end.tv_usec - ref_start.tv_usec) * 1e-6;
    }
    double avg_cpu_time = total_cpu_time / RUNS;

    checkResult(C_naive, C_my, size);

    printf("平均 CPU Time: %lf s\n", avg_cpu_time);
    printf("平均 DCU Time: %lf s\n", avg_dcu_time);
    printf("平均 Speedup (CPU / DCU): %lf\n", avg_cpu_time / avg_dcu_time);

    free(A); free(B); free(C_naive); free(C_my);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}
