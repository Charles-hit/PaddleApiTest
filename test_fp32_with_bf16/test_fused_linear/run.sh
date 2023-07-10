#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
rm new_log_test_fused_linear_develop
# 读取字符串
while IFS= read -r line
do
    # 调用 train.py 并传递字符串作为参数
    python test_fused_linear_develop.py "$line" 2>&1|tee >> new_log_test_fused_linear_develop

done < case.txt
