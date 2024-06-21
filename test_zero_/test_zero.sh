#!/bin/bash
set -x
export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES=2

python test_zero_develop.py
