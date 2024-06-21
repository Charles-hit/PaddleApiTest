#!/bin/bash
set -x
export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES=2

for((i=1;i<=11;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_squared_l2_norm_develop.py TestSquaredl2NormDevelopCase"$i$dtype  
        $cmd
        echo $cmd
    done
done 

