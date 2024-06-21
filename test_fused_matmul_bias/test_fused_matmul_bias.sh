# CUDA_VISABLE_DEVICES=4 python test_fused_linear_develop.py TestFCDevelopCase4_FP32 >> fc.log 2>&1
# CUDA_VISABLE_DEVICES=4 python test_fused_linear_develop.py TestFCDevelopCase4_BFP16 >> fc.log 2>&1
# CUDA_VISABLE_DEVICES=4 python test_fused_linear_develop.py TestFCDevelopCase5_FP32 >> fc.log 2>&1
# CUDA_VISABLE_DEVICES=4 python test_fused_linear_develop.py TestFCDevelopCase5_BFP16 >> fc.log 2>&1
# CUDA_VISABLE_DEVICES=4 python test_fused_linear_develop.py TestFCDevelopCase6_FP32 >> fc.log 2>&1
# CUDA_VISABLE_DEVICES=4 python test_fused_linear_develop.py TestFCDevelopCase6_BFP16 >> fc.log 2>&1
# CUDA_VISABLE_DEVICES=4 python test_fused_linear_develop.py TestFCDevelopCase7_FP32 >> fc.log 2>&1
# CUDA_VISABLE_DEVICES=4 python test_fused_linear_develop.py TestFCDevelopCase7_BFP16 >> fc.log 2>&1

#!/bin/bash
set -x
export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES=1

for((i=8;i<=14;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_fused_matmul_bias_develop.py TestFCDevelopCase"$i$dtype  
        $cmd
        echo $cmd
    done
done 