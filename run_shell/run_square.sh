#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=4

export NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
cd test_square/
for((i=9;i<=16;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_square_develop.py TestSquareDevelopCase"$i$dtype
        $cmd
        echo $cmd
    done
done 
