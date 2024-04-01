import sys
import unittest

import numpy as np
import torch

import paddle
from paddle.utils import map_structure

sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

class TestLinspaceDevelopCase1_FP32(unittest.TestCase):
    def init_dtype(self):
        self.dtype = "float32"

    def init_params(self):
        self.start = 0.0291547594742265
        self.stop = 0.10954451150103323
        self.num = 1000
        self.init_dtype()

    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        _ = self.gen_torch_inputs_and_dout()
        out_torch = self.cal_torch_res(
            self.start, self.stop, self.num
        )
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_np_inputs_and_dout(self):
        # no inputs
        # no grad
        ...

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def gen_torch_inputs_and_dout(self):
        # no inputs
        # no grad
        ...

    def gen_eager_inputs_and_dout(self):
        # no inputs
        # no grad
        ...

    def cal_torch_res(self, start, stop, num):
        out = torch.linspace(start, stop, num)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self, start, stop, num):
        out = paddle.linspace(start, stop, num)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        _ = self.gen_eager_inputs_and_dout()
        out_eager = self.cal_eager_res(
            self.start, self.stop, self.num
        )
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
            api="paddle.linspace",
        )
        # compare develop eager backward res with torch
        # no grad
        ...

class TestLinspaceDevelopCase1_FP16(TestLinspaceDevelopCase1_FP32):
    def init_dtype(self):
        self.dtype = "float16"

class TestLinspaceDevelopCase1_BF16(TestLinspaceDevelopCase1_FP32):
    def init_dtype(self):
        self.dtype = "bfloat16"

if __name__ == '__main__':
    unittest.main()