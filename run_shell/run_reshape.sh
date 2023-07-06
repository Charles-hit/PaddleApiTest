#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=3

export NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
cd  test_reshape
for((i=16;i<=18;i++));  
do
    for dtype in _FP32 _FP16 _BFP16;
    do
        cmd="python test_reshape_develop_new_frl.py TestReshapeDevelopCase"$i$dtype
        $cmd
        echo $cmd
    done
done 
