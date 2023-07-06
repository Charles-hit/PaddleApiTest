#!/bin/bash
set -ex

card_num=$1
version=$2
log_dir=$3

export NVIDIA_TF32_OVERRIDE=0

case $card_num in 
    2 ) export CUDA_VISIBLE_DEVICES=0,1 ;;
    4 ) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
    8 ) export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
    * )  
        echo '请输入正确的卡数'  
        exit 
    ;;
esac

#please run "bash test_vocab_parallel_embedding.sh * develop" first

if [ "$version" == 'develop' ]; then
    # rm -rf data/*.npz
    # python prepare_data.py
    python -m paddle.distributed.launch --log_dir $log_dir get_and_test_paddle_result.py
else
   echo "请输入develop或者incubate"
fi