CUDA_VISABLE_DEVICES=0 python test_layernorm_develop.py TestLayerNormDevelopCase4_FP32 >> layernorm.log 2>&1
CUDA_VISABLE_DEVICES=0 python test_layernorm_develop.py TestLayerNormDevelopCase4_BFP16 >> layernorm.log 2>&1