module purge

module load compiler/devtoolset/7.3.1 # （如果需要 GNU 工具链）
module load compiler/dtk/24.04          # DTK 24.04 环境

hipcc GEMM1.cpp -o GEMM1 -lrocblas