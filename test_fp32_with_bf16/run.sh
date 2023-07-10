!/bin/bash
for dir in test_flash_attention/ test_fused_linear/ test_layernorm/ test_matmul/ test_silu/ test_vocab_parallel_embedding/
do
   cd $dir
   ./run.sh &
   cd ..
done
