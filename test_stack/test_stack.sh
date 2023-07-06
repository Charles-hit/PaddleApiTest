CUDA_VISABLE_DEVICES=1 python test_stack_develop.py TestStackDevelopCase2_FP32 >> stack.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_stack_develop.py TestStackDevelopCase2_BFP16 >> stack.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_stack_develop.py TestStackDevelopCase3_FP32 >> stack.log 2>&1
CUDA_VISABLE_DEVICES=1 python test_stack_develop.py TestStackDevelopCase3_BFP16 >> stack.log 2>&1