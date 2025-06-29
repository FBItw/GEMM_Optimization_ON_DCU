#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <omp.h>
#include <math.h>

#define N 4096
#define TILE 16
#define REPEAT 20

__global__ void myGEMM_optimized(double *A, double *B, double *C, int width) {
    __shared__ double tile_A[TILE][TILE];
    __shared__ double tile_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0.0;
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

void checkResult(double *hostRef, double *gpuRef, int n) {
    double epsilon = 1e-4;
    double max_rel_error = 0.0, sum_rel_error = 0.0;
    bool flag = true;
    for (int i = 0; i < n; i++) {
        double diff = fabs(hostRef[i] - gpuRef[i]);
        double base = fabs(hostRef[i]) > 1e-6 ? fabs(hostRef[i]) : 1.0;
        double rel_err = diff / base;
        if (rel_err > epsilon) {
            flag = false;
        }
        max_rel_error = fmax(max_rel_error, rel_err);
        sum_rel_error += rel_err;
    }
    if (flag) {
        printf("计算正确 ✅\n");
    } else {
        printf("计算错误 ❌\n");
        printf("最大相对误差: %e\n", max_rel_error);
        printf("平均相对误差: %e\n", sum_rel_error / n);
    }
}

int main() {
    int size = N * N;
    size_t bytes = size * sizeof(double);

    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C_naive = (double*)malloc(bytes);
    double *C_my = (double*)malloc(bytes);

    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);

    srand(12345);
    for (int i = 0; i < size; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // === DCU 计时 ===
    double total_dcu_time = 0.0;
    for (int t = 0; t < REPEAT; ++t) {
        hipMemset(d_C, 0, bytes);
        struct timeval mystart, myend;
        gettimeofday(&mystart, NULL);
        hipLaunchKernelGGL(myGEMM_optimized, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, N);
        hipDeviceSynchronize();
        gettimeofday(&myend, NULL);
        total_dcu_time += (myend.tv_sec - mystart.tv_sec) + (myend.tv_usec - mystart.tv_usec) * 1.e-6;
    }
    hipMemcpy(C_my, d_C, bytes, hipMemcpyDeviceToHost);
    double avg_dcu_time = total_dcu_time / REPEAT;

    // === CPU 计时 ===
    double total_cpu_time = 0.0;
    for (int t = 0; t < REPEAT; ++t) {
        struct timeval ref_start, ref_end;
        gettimeofday(&ref_start, NULL);
#pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C_naive[i * N + j] = sum;
            }
        gettimeofday(&ref_end, NULL);
        total_cpu_time += (ref_end.tv_sec - ref_start.tv_sec) + (ref_end.tv_usec - ref_start.tv_usec) * 1.e-6;
    }
    double avg_cpu_time = total_cpu_time / REPEAT;

    checkResult(C_naive, C_my, size);

    printf("平均 CPU Time: %lf s\n", avg_cpu_time);
    printf("平均 DCU Time: %lf s\n", avg_dcu_time);
    printf("平均加速比 (CPU / DCU): %lf\n", avg_cpu_time / avg_dcu_time);

    free(A); free(B); free(C_naive); free(C_my);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}
