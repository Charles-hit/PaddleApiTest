#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=5

export NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
cd test_add_inplace/
for((i=3;i<=10;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_add_inplace_develop.py TestAddInplaceDevelopCase"$i$dtype
        $cmd
        echo $cmd
    done
done 
