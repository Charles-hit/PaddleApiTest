import numpy as np
import init_config_class

dim_1_list = [31250]
dim_2_list = [8192]
dim_3_list = [14336]

card_num = 8

def generate_np_inputs_and_dout():

    for i in range(8):
        np.random.seed(0)

        dim_1 = dim_1_list[0]
        dim_2 = dim_2_list[0]
        dim_3 = dim_3_list[0]
        x_case1 = np.random.randint(low=0, high=dim_1, size=[1, dim_2]).astype("int64")
        table_case1 = np.random.random(size=[dim_1, dim_3]).astype("float32") - 0.5 
        dout_case1 = np.random.random(size=[1, dim_2, dim_3]).astype("float32") - 0.5

        np_input_dir_forward = "./data/vpe-int64-bf16-bf16-eager_in_tmp_477-word_embedding_expanded_{rank}.w_0-eager_in_tmp_486-pp-0-mp-{rank}.npz".format(rank=i)
        np_input_dir_backward = "./data/vpe-int64-bf16-bf16-eager_in_tmp_477-word_embedding_expanded_{rank}.w_0-eager_in_tmp_486-pp-0-mp-{rank}.npy".format(rank=i)

        np.savez(np_input_dir_forward, x = x_case1, weight = table_case1, num_embeddings=dim_1 * card_num, embedding_dim=dim_3)
        np.save(np_input_dir_backward, dout_case1)


generate_np_inputs_and_dout()