# CUDA_VISABLE_DEVICES=1 python test_matmul_develop.py TestMatmulDevelopCase10_FP32 >> matmul.log 2>&1
# CUDA_VISABLE_DEVICES=1 python test_matmul_develop.py TestMatmulDevelopCase10_BFP16 >> matmul.log 2>&1
UDA_VISABLE_DEVICES=1 python test_matmul_develop.py TestMatmulDevelopCase11_FP32 >> matmul_new.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_matmul_develop.py TestMatmulDevelopCase11_BFP16 >> matmul_new.log 2>&1