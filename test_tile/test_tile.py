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


class TestTileDevelopCase1_FP32(unittest.TestCase):
    def init_dtype(self):
        self.dtype = "float32"

    def init_params(self):
        self.x_shape = [10, 4, 1, 40, 72]
        self.repeat_times = [1, 1, 1, 1, 1]
        self.dout_shape = [10, 4, 1, 40, 72]
        self.init_dtype()

    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, self.repeat_times, dout_torch
        )
        del x_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.detach().cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_np_inputs_and_dout(self):
        self.np_x = np.random.random(size=self.x_shape).astype("float32") - 0.5
        self.np_dout = (
            np.random.random(size=self.dout_shape).astype("float32") - 0.5
        )
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
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
        return x_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, dout_eager

    def cal_torch_res(self, x, repeat_times, dout):
        x_t = x
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        out = torch.tile(x_t, repeat_times)
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, x, repeat_times, dout):
        x_t = x
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.tile(x_t, repeat_times)
        out_grads = paddle.grad([out], [x], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, self.repeat_times, dout_eager
        )
        del x_eager
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
            api="paddle.tile",
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
                api="paddle.tile",
            )

class TestTileDevelopCase1_FP16(TestTileDevelopCase1_FP32):
    def init_dtype(self):
        self.dtype = "float16"

class TestTileDevelopCase1_BF16(TestTileDevelopCase1_FP32):
    def init_dtype(self):
        self.dtype = "bfloat16"

if __name__ == '__main__':
    unittest.main()