#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cmath>

// 随机初始化 double 类型矩阵 [0, 10)
void init_random_double_matrix(double* mat, size_t size, double min_val = 0.0, double max_val = 10.0) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (size_t i = 0; i < size; ++i) {
        mat[i] = dist(rng);
    }
}

// 随机选取矩阵维度值
int pick_random_size(std::mt19937& rng) {
    std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048, 4096};
    std::uniform_int_distribution<int> dist(0, sizes.size() - 1);
    return sizes[dist(rng)];
}

int main(int argc, char* argv[]) {
    int T = 4; // 默认运行次数
    if (argc > 1) {
        try {
            T = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "输入无效，使用默认 T=4\n";
        }
    }

    std::mt19937 rng(std::random_device{}());
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    for (int t = 0; t < T; ++t) {
        int M = pick_random_size(rng);
        int K = pick_random_size(rng);
        int N = pick_random_size(rng);

        size_t sizeA = M * K;
        size_t sizeB = K * N;
        size_t sizeC = M * N;

        double *A_d, *B_d, *C_d;
        hipMalloc(&A_d, sizeA * sizeof(double));
        hipMalloc(&B_d, sizeB * sizeof(double));
        hipMalloc(&C_d, sizeC * sizeof(double));

        std::vector<double> A_h(sizeA);
        std::vector<double> B_h(sizeB);
        std::vector<double> C_h(sizeC);
        std::vector<double> C_ref(sizeC, 0.0);

        init_random_double_matrix(A_h.data(), sizeA);
        init_random_double_matrix(B_h.data(), sizeB);

        hipMemcpy(A_d, A_h.data(), sizeA * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(B_d, B_h.data(), sizeB * sizeof(double), hipMemcpyHostToDevice);

        double alpha = 1.0, beta = 0.0;

        auto start = std::chrono::high_resolution_clock::now();

        rocblas_dgemm(handle,
                      rocblas_operation_none, rocblas_operation_none,
                      M, N, K,
                      &alpha,
                      A_d, M,
                      B_d, K,
                      &beta,
                      C_d, M);

        hipDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        hipMemcpy(C_h.data(), C_d, sizeC * sizeof(double), hipMemcpyDeviceToHost);

        std::cout << "Case " << t + 1 << ": GEMM [" << M << "x" << K << "] × [" << K << "x" << N << "] = [" << M << "x" << N << "]\n";
        std::cout << "Time: " << elapsed << " ms" << std::endl;

        // CPU 端计算 col-major 的 C_ref
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                double sum = 0.0;
                for (int k = 0; k < K; ++k) {
                    sum += A_h[k * M + i] * B_h[j * K + k];
                }
                C_ref[j * M + i] = sum;
            }
        }

        // 检查最多前 1000 项误差
        bool correct = true;
        size_t check_num = std::min(size_t(1000), sizeC);
        double tol = (M * N <= 512 * 512) ? 5e-6 : 1e-4;

        for (size_t i = 0; i < check_num; ++i) {
            double diff = std::abs(C_ref[i] - C_h[i]);
            double denom = std::max(1.0, std::abs(C_ref[i]));
            if (diff > tol * denom) {
                correct = false;
                std::cout << "Mismatch at index " << i
                          << ": ref=" << C_ref[i]
                          << ", val=" << C_h[i]
                          << ", diff=" << diff << "\n";
                break;
            }
        }

        std::cout << (correct ? "计算正确 ✅\n\n" : "计算错误 ❌\n\n");

        hipFree(A_d); hipFree(B_d); hipFree(C_d);
    }

    rocblas_destroy_handle(handle);
    return 0;
}
