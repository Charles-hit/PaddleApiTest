import numpy as np

dim_1 = 4096
dim_2 = 12288
dim_3 = 8192
dim_4 = 14336

def generate_np_inputs_and_dout():
    x_case1 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5
    dout_case1 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5

    x_case2 = np.random.random(size=[1, dim_2]).astype("float32") - 0.5
    dout_case2 = np.random.random(size=[1, dim_2]).astype("float32") - 0.5

    x_case3 = np.random.random(size= [1, dim_3, dim_4]).astype("float32") - 0.5
    dout_case3 = np.random.random(size= [1, dim_3, dim_4]).astype("float32") - 0.5

    np.savez("./inputs_case1.npz", x = x_case1, dout = dout_case1)
    np.savez("./inputs_case2.npz", x = x_case2, dout = dout_case2)
    np.savez("./inputs_case3.npz", x = x_case3, dout = dout_case3)


generate_np_inputs_and_dout()