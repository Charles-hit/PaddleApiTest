rm -rf ./data
rm -rf ./log
rm -rf ./torch_value
rm -rf ./paddle_value
rm -f ./out_torch.log ./out_paddle.log
cases=(0)
for case in ${cases[@]}
do
    echo "Testing Case $case"
    python gen_np_inputs.py $case
    torchrun --nproc-per-node=8 run_torch.py $case 1>>out_torch.log 2>>out_torch.log
    python -m paddle.distributed.launch run__c_softmax_with_cross_entropy.py $case 1>>out_paddle.log 2>>out_paddle.log
    python check.py $case
done