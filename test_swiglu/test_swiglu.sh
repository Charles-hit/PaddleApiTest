#!/bin/bash
set -x
export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES=2

for((i=1;i<=1;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_swiglu_develop.py TestSwigluDevelopCase"$i$dtype  
        $cmd
        echo $cmd
    done
done 

