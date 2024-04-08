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

class TestCastDevelopCase1_FP32(unittest.TestCase):
    def set_up(self):
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
        self.shape = [4096, 4096, 256]
    
    def init_threshold(self):
        self.atol = TOLERANCE[self.in_dtype]["atol"]
        self.rtol = TOLERANCE[self.in_dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=self.shape).astype("float32") - 0.5
        # convert np array dtype
        if self.in_dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.in_dtype),
            # if self.in_dtype != 'bfloat16'
            # else torch.float32,
            requires_grad=False,
        )
        return x_torch
    
    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.in_dtype if self.in_dtype != 'bfloat16' else "uint16",
            place="gpu",
        )
        x_eager.stop_gradient = True
        return x_eager

    def cal_torch_res(self, x):
        # if self.in_dtype == "bfloat16":
        #     x = x.to(dtype=torch.bfloat16)
        out = x.to(convert_dtype_to_torch_type(self.out_dtype))
        if self.out_dtype == "bfloat16":
            out = out.to(dtype=torch.float16)
        return out

    def cal_eager_res(self, x):
        # if self.in_dtype == "bfloat16":
        #     x = paddle.cast(x, dtype="uint16")
        out = paddle.cast(x, self.out_dtype)
        if self.out_dtype == "bfloat16":
            out = paddle.cast(out, dtype="float16")
        return out


    def test_eager_accuracy(self):
        for in_dtype in ["bfloat16"]:
            for out_dtype in ["float32", "float16", "bfloat16"]:
                if in_dtype != out_dtype:
                    self.in_dtype = in_dtype
                    self.out_dtype = out_dtype
                    self.set_up()
                    self.run_check()
                    print(f"cast {in_dtype} to {out_dtype} finish check.")

    def run_check(self):
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
            self.in_dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="eager",
            fwd_or_bkd="forward",
            api="paddle.cast",
        )

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()