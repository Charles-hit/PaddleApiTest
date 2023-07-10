#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
# 读取字符串
rm new_log_matmul_fp32vsbf16
while IFS= read -r line
do
    # 调用 train.py 并传递字符串作为参数
    python test_matmul_fp32vsbfp16.py "$line" 2>&1|tee >> new_log_matmul_fp32vsbf16
done < case.txt
