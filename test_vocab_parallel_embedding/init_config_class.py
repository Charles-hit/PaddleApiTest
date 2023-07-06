import numpy as np
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

card_num = 8

dim_1 = [56200, 31250 * card_num]
dim_2 = [4096, 8192]
dim_3 = [12288, 14336]

class InitConfigClass:
    def __init__(self):
        self._init_params()
        self._init_threshold()
        self._init_np_inputs_and_dout()

    def _init_params(self, np_input_dir="", dtype="", save_static_res_path="" , save_eager_res_path=""):
        self._np_input_dir = np_input_dir
        self._dtype = dtype
        self._save_static_res_path = save_static_res_path
        self._save_eager_res_path = save_eager_res_path
    
    def _init_threshold(self):
        self._atol = TOLERANCE[self._dtype]["atol"]
        self._rtol = TOLERANCE[self._dtype]["rtol"]

    def _init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self._np_input_dir)
        # get np array from npz file
        self._np_x = np_inputs_array["x"]
        self._np_table = np_inputs_array["table"]
        self._np_dout = np_inputs_array["dout"]
        # convert np array dtype
        if self._dtype == "float16":
            self._np_table = self._np_table.astype("float16")
            self._np_dout = self._np_dout.astype("float16")