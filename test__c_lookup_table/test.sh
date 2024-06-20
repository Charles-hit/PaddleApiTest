rm -rf ./data
rm -rf ./log
rm -rf ./torch_value
rm -rf ./paddle_value
rm -f ./out_torch.log ./out_paddle.log
cases=(0 1 2)
for case in ${cases[@]}
do
    echo "Testing Case $case"
    python gen_np_inputs.py $case
    python run_torch.py $case 1>>out_torch.log 2>>out_torch.log
    python -m paddle.distributed.launch run__c_lookup_table.py $case 1>>out_paddle.log 2>>out_paddle.log
    python check.py $case
done