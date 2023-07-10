#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
rm new_log_test_layernorm_fp32vsbfp16
# 读取字符串
while IFS= read -r line
do
    # 调用 train.py 并传递字符串作为参数
    python test_layernorm_fp32vsbfp16.py "$line" 2>&1|tee >> new_log_test_layernorm_fp32vsbfp16

done < case.txt
