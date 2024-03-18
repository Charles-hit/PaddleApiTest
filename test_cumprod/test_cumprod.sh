CUDA_VISABLE_DEVICES=1 python test_cumprod_develop.py TestCumprodDevelopCase1_FP32 > cumprod_new.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_cumprod_develop.py TestCumprodDevelopCase1_FP16 >> cumprod_new.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_cumprod_develop.py TestCumprodDevelopCase1_BF16 >> cumprod_new.log 2>&1