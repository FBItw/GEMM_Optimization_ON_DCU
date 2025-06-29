# GEMM_Optimization_ON_DCU
Overview
  Introduction to Competition Tasks:
  Task 1: Matrix multiplication operation. According to the given matrix, solve matrix multiplication of different scales and output the final result Calculate the       results and the required time.
  Task 2: Based on this, efficient matrix multiplication operations can be achieved using program performance optimization theories and methods such as memory access     optimization, partitioning techniques, register optimization, loop unrolling, SIMD vectorization, and data prefetching.
  Task 3: Sparse matrix acceleration algorithm optimization. Design efficient data compression formats, sparse scheduling strategies, and parallelization strategies      for matrices with different sparsity rates and structures.
Additional Question 1: Matrix Transposition.
  Runtime Environment
  1.Hardware Environment: CPU: Intel 8458P, with 20 available CPU cores. One DCU heterogeneous accelerator (k100AI) with 64GB of VRAM.
  2.Software Environment: hip5.4.23416, devtoolset-7.3.1, dtk-24.04, rocSPARSE library.
Execution Steps
  1.Modify the compilation script and the job submission script .build.shdcutest.slurm
  2.Execute the command to compile.bash build.sh
  3.Execute the command to submit the job.sbatch dcutest.slurm
  4.Check the generated file in the directory to view the runtime results..outout
  5.NOTE:If modifying the number of matrix groups in Task 1, it needs to be done in the last row of. slurm /Change the number 10 to the desired number of matrix        groups in GEMM1 10. If you need to modify the dimensions of the matrix, you need to add the required dimension, such as 32, in the twentieth row of GEMM1.cpp. If     you need to modify the matrix dimension in task 2, you need to modify line 9 # define N 4096 in GEMM2.cpp. If you want to change it to a 2048 square matrix, you      only need to modify 4096 to 2048.
