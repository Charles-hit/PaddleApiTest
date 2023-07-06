import numpy as np
import sys
sys.path.append("../../..")
from utils import TOLERANCE, convert_dtype_to_torch_type

class InitConfigClass:
    def __init__(self):
        self._init_params()
        self._init_threshold()
        self._init_np_inputs_and_dout()

    def _init_params(self, np_input_dir_forward="", np_input_dir_backward="", dtype=""):
        self._np_input_dir_forward = np_input_dir_forward
        self._np_input_dir_backward = np_input_dir_backward
        self._dtype = dtype
    
    def _init_threshold(self):
        self._atol = TOLERANCE[self._dtype]["atol"]
        self._rtol = TOLERANCE[self._dtype]["rtol"]

    def _init_np_inputs_and_dout(self):
        np_inputs_array_forward = np.load(self._np_input_dir_forward)
        np_inputs_array_backward = np.load(self._np_input_dir_backward)
        # get np array from npz file
        self._np_x = np_inputs_array_forward["x"]
        self._np_table = np_inputs_array_forward["weight"]
        self._np_dout = np_inputs_array_backward
        self._num_embeddings = np_inputs_array_forward['num_embeddings']
        self._embedding_dim = np_inputs_array_forward['embedding_dim']