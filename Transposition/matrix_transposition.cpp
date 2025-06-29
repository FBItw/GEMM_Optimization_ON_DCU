#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <hip/hip_runtime.h>

using namespace std;

//生成随机矩阵
vector<int8_t> generateMatrix(int n) {
    vector<int8_t> matrix(n * n);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int8_t> dist(-128, 127);
    for (int i = 0; i < n * n; ++i)
       matrix[i] = dist(gen);
    return matrix;
}

// 串行矩阵转置
vector<int8_t> serialTranspose(const vector<int8_t>& matrix, int n) {
    vector<int8_t> result(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[j * n + i] = matrix[i * n + j];
        }
    }
    return result;
}

// 棋盘划分并行转置
vector<int8_t> checkerboardTranspose(const vector<int8_t>& matrix, int n, int p) {
    vector<int8_t> result(n * n);
    // 计算块大小，并对齐到64字节以减少假共享
    int block_size = (n + p - 1) / p;
    block_size = ((block_size + 63) / 64) * 64;  
    if (block_size < 64) block_size = n < 64 ? n : 64;
    // 使用collapse展开两层循环，限制线程数为p，绑定线程以提高NUMA局部性
    #pragma omp parallel for collapse(2) schedule(static) num_threads(p) proc_bind(close)
    for (int bi = 0; bi < p; ++bi) {
        for (int bj = 0; bj < p; ++bj) {
            int start_x = bi * block_size;
            int end_x   = std::min((bi + 1) * block_size, n);
            int start_y = bj * block_size;
            int end_y   = std::min((bj + 1) * block_size, n);
            for (int y = start_y; y < end_y; ++y) {
                for (int x = start_x; x < end_x; ++x) {
                    result[y * n + x] = matrix[x * n + y];
                }
            }
        }
    }
    return result;
}

vector<int8_t> rectangularTranspose(const vector<int8_t>& matrix, int n, int p) {
    vector<int8_t> result(n * n);
    int rows_per_thread = 256;
    if (rows_per_thread > n) rows_per_thread = n;
    const int8_t* data = matrix.data();
    #pragma omp parallel for schedule(static) num_threads(p) proc_bind(close)
    for (int ti = 0; ti < p; ++ti) {
        int start_row = ti * rows_per_thread;
        int end_row = std::min((ti + 1) * rows_per_thread, n);
        for (int y = 0; y < n; ++y) {
            for (int x = start_row; x < end_row; ++x) {
                result[y * n + x] = data[x * n + y];
            }
        }
    }
    return result;
}

// 利用共享内存的GPU矩阵转置kernel
__global__ void gpuTransposeKernel(const int8_t* input, int8_t* output, int n) {
    __shared__ int8_t tile[32][32 + 1];  // 32x32 tile，+1避免共享内存Bank冲突
    unsigned int xIndex = blockIdx.x * 32 + threadIdx.x;
    unsigned int yIndex = blockIdx.y * 32 + threadIdx.y;
    if (xIndex < n && yIndex < n) {
        // 将全局内存的数据读取到共享内存tile
        tile[threadIdx.y][threadIdx.x] = input[yIndex * n + xIndex];
    }
    __syncthreads();
    unsigned int transposed_x = blockIdx.y * 32 + threadIdx.x;
    unsigned int transposed_y = blockIdx.x * 32 + threadIdx.y;
    if (transposed_x < n && transposed_y < n) {
        output[transposed_y * n + transposed_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// GPU矩阵转置接口函数
vector<int8_t> hipTranspose(const vector<int8_t>& matrix, int n) {
    size_t bytes = n * n * sizeof(int8_t);
    int8_t *d_input = nullptr, *d_output = nullptr;
    hipMalloc(&d_input, bytes);
    hipMalloc(&d_output, bytes);

    hipMemcpy(d_input, matrix.data(), bytes, hipMemcpyHostToDevice);

    // 定义线程块和网格维度，每个线程块处理32x32的tile
    dim3 blockSize(32, 32);
    dim3 gridSize((n + 31) / 32, (n + 31) / 32);
    hipLaunchKernelGGL(gpuTransposeKernel, gridSize, blockSize, 0, 0, d_input, d_output, n);
    hipMemcpy(nullptr, nullptr, 0, hipMemcpyHostToHost);  

    // 将结果拷回主机内存
    vector<int8_t> result(n * n);
    hipMemcpy(result.data(), d_output, bytes, hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
    return result;
}

int main() {
    //cout << "矩阵转置算法对比实验\n";
    int n=25000;
    //cout << "输入矩阵维度 n (n > 0): ";
    //cin >> n;
    auto matrix = generateMatrix(n);
    int max_threads = omp_get_max_threads();
    int p=37;
    //cout << "请输入线程数 (1 - " << max_threads << "): ";
    //cin >> p;
    auto start = chrono::high_resolution_clock::now();
    auto serial_result = serialTranspose(matrix, n);
    auto serial_time = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cout << "串行 CPU 耗时: " << serial_time << " 秒\n";

    start = chrono::high_resolution_clock::now();
    auto checkerboard_result = checkerboardTranspose(matrix, n, p);
    auto checkerboard_time = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cout << "棋盘划分并行 CPU 耗时: " << checkerboard_time << " 秒\n";

    start = chrono::high_resolution_clock::now();
    auto rectangular_result = rectangularTranspose(matrix, n, p);
    auto rectangular_time = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cout << "直角划分并行 CPU 耗时: " << rectangular_time << " 秒\n";

    start = chrono::high_resolution_clock::now();
    auto hip_result = hipTranspose(matrix, n);
    auto hip_time = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();
    cout << "HIP GPU 耗时: " << hip_time << " 秒\n";
    cout << "\n性能对比:\n";
    cout << "算法\t\t\t时间(秒)\t\t加速比\n";
    cout << "串行 CPU\t\t" << serial_time << "\t1.0\n";
    cout << "棋盘划分并行 CPU\t" << checkerboard_time << "\t" << serial_time / checkerboard_time << "\n";
    cout << "直角划分并行 CPU\t" << rectangular_time << "\t" << serial_time / rectangular_time << "\n";
    cout << "HIP GPU\t\t\t" << hip_time << "\t" << serial_time / hip_time << "\n";
    return 0;
}
