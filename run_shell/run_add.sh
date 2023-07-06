#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
cd test_add
#for((i=9;i<=19;i++));  
for((i=20;i<=20;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_add_develop.py TestAddDevelopCase"$i$dtype
        $cmd
        echo $cmd
    done
done 
