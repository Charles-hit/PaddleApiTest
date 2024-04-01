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

class TestWhereDevelopCase1_FP32(unittest.TestCase):
    def init_dtype(self):
        self.dtype = "float32"

    def init_params(self):
        self.condition_shape = [10, 1, 256, 256]
        self.x_shape = [10, 1, 256, 256]
        self.y_shape = [10, 1, 256, 256]
        self.dout_shape = [10, 1, 256, 256]
        self.init_dtype()

    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        condition_torch, x_torch, y_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            condition_torch, x_torch, y_torch, dout_torch
        )
        del condition_torch
        del x_torch
        del y_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_np_inputs_and_dout(self):
        self.np_condition = np.random.random(size=self.x_shape) < 0.5
        self.np_x = np.random.random(size=self.x_shape).astype("float32") - 0.5
        self.np_y = np.random.random(size=self.y_shape).astype("float32") - 0.5
        self.np_dout = (
            np.random.random(size=self.dout_shape).astype("float32") - 0.5
        )
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_y = self.np_y.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def gen_torch_inputs_and_dout(self):
        condition_torch = torch.tensor(
            self.np_condition,
            device='cuda',
            dtype=torch.bool,
            requires_grad=False,
        )
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        y_torch = torch.tensor(
            self.np_y,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return condition_torch, x_torch, y_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        condition_eager = paddle.to_tensor(
            self.np_condition,
            dtype="bool",
            place="gpu",
        )
        condition_eager.stop_gradient = True
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        y_eager = paddle.to_tensor(
            self.np_y,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        y_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return condition_eager, x_eager, y_eager, dout_eager

    def cal_torch_res(self, condition, x, y, dout):
        condition_t = condition
        x_t = x
        y_t = y
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            y_t = y.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        out = torch.where(condition_t, x_t, y_t)
        out_grads = torch.autograd.grad([out], [x, y], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, condition, x, y, dout):
        condition_t = condition
        x_t = x
        y_t = y
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            y_t = paddle.cast(y, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.where(condition_t, x_t, y_t)
        out_grads = paddle.grad([out], [x, y], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        condition_eager, x_eager, y_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            condition_eager, x_eager, y_eager, dout_eager
        )
        del condition_eager
        del x_eager
        del y_eager
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        del out_eager
        del out_grads_eager
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
            api="paddle.where",
        )
        # compare develop eager backward res with torch
        for idx in range(len(out_grads_eager_np)):
            np_assert_accuracy(
                out_grads_eager_np[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="backward",
                api="paddle.where",
            )

class TestWhereDevelopCase1_FP16(TestWhereDevelopCase1_FP32):
    def init_dtype(self):
        self.dtype = "float16"

class TestWhereDevelopCase1_BF16(TestWhereDevelopCase1_FP32):
    def init_dtype(self):
        self.dtype = "bfloat16"

if __name__ == '__main__':
    unittest.main()