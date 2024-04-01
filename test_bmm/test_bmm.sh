CUDA_VISABLE_DEVICES=1 python test_bmm_develop.py TestBmmDevelopCase1_FP32 > bmm_new.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_bmm_develop.py TestBmmDevelopCase1_FP16 >> bmm_new.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_bmm_develop.py TestBmmDevelopCase1_BF16 >> bmm_new.log 2>&1