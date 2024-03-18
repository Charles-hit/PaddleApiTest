import sys
import unittest

import numpy as np
import torch

import paddle

sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
)

class TestZero_DevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch = self.gen_torch_inputs_and_dout()
        out_torch = self.cal_torch_res(x_torch)
        del x_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.size = [1536]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=self.size).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32
        )
        return x_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        return x_eager

    def cal_torch_res(self, x):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
        x.zero_()
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.float32)
        return x

    def cal_eager_res(self, x):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
        x.zero_()
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="float32")
        return x

    def test_eager_accuracy(self):
        x_eager = self.gen_eager_inputs_and_dout()
        out_eager = self.cal_eager_res(x_eager)
        del x_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        del out_eager
        paddle.device.cuda.empty_cache()
        # compare develop eager forward res with torch
        np_assert_accuracy(
            out_eager_np,
            self.out_torch,
            self.atol,
            self.rtol,
            self.dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="eager",
            fwd_or_bkd="forward",
            api="paddle.zero_",
        )


class TestZero_DevelopCase1_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1536]

class TestZero_DevelopCase1_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [1536]

class TestZero_DevelopCase2_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [1536, 4096]

class TestZero_DevelopCase2_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1536, 4096]

class TestZero_DevelopCase2_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [1536, 4096]

class TestZero_DevelopCase3_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [4]

class TestZero_DevelopCase3_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4]

class TestZero_DevelopCase3_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [4]

class TestZero_DevelopCase4_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [4096, 1536]

class TestZero_DevelopCase4_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4096, 1536]

class TestZero_DevelopCase4_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [4096, 1536]


class TestZero_DevelopCase5_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [3072, 1536]

class TestZero_DevelopCase5_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [3072, 1536]

class TestZero_DevelopCase5_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [3072, 1536]

class TestZero_DevelopCase6_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [1536, 1536]

class TestZero_DevelopCase6_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1536, 1536]

class TestZero_DevelopCase6_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [1536, 1536]


class TestZero_DevelopCase7_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [4, 4, 3, 3, 3]

class TestZero_DevelopCase7_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4, 4, 3, 3, 3]

class TestZero_DevelopCase7_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [4, 4, 3, 3, 3]


class TestZero_DevelopCase8_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [1536, 4, 1, 4, 4]

class TestZero_DevelopCase8_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1536, 4, 1, 4, 4]

class TestZero_DevelopCase8_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [1536, 4, 1, 4, 4]

class TestZero_DevelopCase9_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [1536, 64]

class TestZero_DevelopCase9_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1536, 64]

class TestZero_DevelopCase9_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [1536, 64]


class TestZero_DevelopCase10_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [1536, 4608]

class TestZero_DevelopCase10_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1536, 4608]

class TestZero_DevelopCase10_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [1536, 4608]

class TestZero_DevelopCase11_FP32(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [64]

class TestZero_DevelopCase11_FP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [64]

class TestZero_DevelopCase11_BFP16(TestZero_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [64]



if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()