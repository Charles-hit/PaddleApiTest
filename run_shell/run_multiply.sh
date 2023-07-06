#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=1

export NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
cd test_multiply/
#for((i=4;i<=8;i++));  
#for((i=5;i<=8;i++));  
for((i=8;i<=8;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_develop_multiply.py TestMultiplyDevelopCase"$i$dtype
        $cmd
        echo $cmd
    done
done 
