CUDA_VISABLE_DEVICES=2 python test_stride_slice_develop.py TestStrideSliceDevelopCase2_FP32 >>stride_slice.log 2>&1
CUDA_VISABLE_DEVICES=2 python test_stride_slice_develop.py TestStrideSliceDevelopCase2_BFP16 >>stride_slice.log 2>&1
CUDA_VISABLE_DEVICES=2 python test_stride_slice_develop.py TestStrideSliceDevelopCase3_FP32 >>stride_slice.log 2>&1
CUDA_VISABLE_DEVICES=2 python test_stride_slice_develop.py TestStrideSliceDevelopCase3_BFP16 >>stride_slice.log 2>&1