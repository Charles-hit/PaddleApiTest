CUDA_VISABLE_DEVICES=1 python test_clip_develop.py TestClipDevelopCase1_FP32 > clip_new.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_clip_develop.py TestClipDevelopCase1_FP16 >> clip_new.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_clip_develop.py TestClipDevelopCase1_BF16 >> clip_new.log 2>&1