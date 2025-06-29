#!/bin/bash
#BATCH -o %j.out
#SBATCH -J test_wcx
#SBATCH -p wzidnormal
#SBATCH --gres=dcu:1		#指定每个节点使用1块DCU卡
#SBATCH -w xdb5



module purge
module load compiler/devtoolset/7.3.1
module load compiler/dtk/24.04

# 执行程序
./GEMM2