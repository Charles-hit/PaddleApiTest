import numpy as np
import init_config_class

def generate_np_inputs_and_dout():
    np.random.seed(0)
    dim_1 = init_config_class.dim_1[0]
    dim_2 = init_config_class.dim_2[0]
    dim_3 = init_config_class.dim_3[0]
    x_case1 = np.random.randint(low=0, high=dim_1, size=[1, dim_2]).astype("int64")
    table_case1 = np.random.random(size=[dim_1, dim_3]).astype("float32") - 0.5 
    dout_case1 = np.random.random(size=[1, dim_2, dim_3]).astype("float32") - 0.5

    np.savez("./inputs_case1.npz", x = x_case1, table = table_case1, dout = dout_case1)

    dim_1 = init_config_class.dim_1[1]
    dim_2 = init_config_class.dim_2[1]
    dim_3 = init_config_class.dim_3[1]
    x_case1 = np.random.randint(low=0, high=dim_1, size=[1, dim_2]).astype("int64")
    table_case1 = np.random.random(size=[dim_1, dim_3]).astype("float32") - 0.5 
    dout_case1 = np.random.random(size=[1, dim_2, dim_3]).astype("float32") - 0.5

    np.savez("./inputs_case2.npz", x = x_case1, table = table_case1, dout = dout_case1)