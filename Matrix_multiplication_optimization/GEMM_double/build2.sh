#!/bin/bash
rm -f GEMM2

module purge
module load compiler/devtoolset/7.3.1 # （如果需要 GNU 工具链）
module load compiler/dtk/24.04          # DTK 24.04 环境

hipcc GEMM2.cpp -O3 --offload-arch=gfx908 -ffast-math -o GEMM2

if [ $? -eq 0 ]; then
  echo "✅ 编译成功: GEMM2 已生成"
else
  echo "❌ 编译失败"
fi