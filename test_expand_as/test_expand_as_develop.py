# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

int_types = ["bool"]


class TestExpandDevelopCase1_FP32(unittest.TestCase):
    def init_dtype(self):
        self.dtype = "bool"

    def init_params(self):
        self.x_shape = [1024, 1]
        self.param_shape = [1024, 8]
        self.dout_shape = [1024, 8]
        self.init_dtype()

    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, y_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(x_torch, y_torch, dout_torch)
        del x_torch
        del y_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().detach().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_np_inputs_and_dout(self):
        self.np_x = np.random.random(size=self.x_shape).astype("float32") - 0.5
        self.np_y = np.random.random(size=self.param_shape).astype("float32") - 0.5
        self.np_dout = np.random.random(size=self.dout_shape).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_y = self.np_y.astype("float16")
            self.np_dout = self.np_dout.astype("float16")
        elif self.dtype == "bool":
            self.np_x = self.np_x > 0
            self.np_dout = None

    def init_threshold(self):
        if self.dtype in int_types:
            return
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device="cuda",
            dtype=(
                convert_dtype_to_torch_type(self.dtype)
                if self.dtype != "bfloat16"
                else torch.float32
            ),
            requires_grad=self.dtype not in int_types,
        )
        y_torch = torch.tensor(
            self.np_y,
            device="cuda",
            dtype=convert_dtype_to_torch_type(self.dtype),
            requires_grad=False,
        )
        if self.dtype in int_types:
            return x_torch, y_torch, None
        dout_torch = torch.tensor(
            self.np_dout,
            device="cuda",
            dtype=(
                convert_dtype_to_torch_type(self.dtype)
                if self.dtype != "bfloat16"
                else torch.float32
            ),
            requires_grad=self.dtype not in int_types,
        )
        return x_torch, y_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = self.dtype in int_types
        y_eager = paddle.to_tensor(
            self.np_y,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            place="gpu",
        )
        if self.dtype in int_types:
            return x_eager, y_eager, None
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = self.dtype in int_types
        return x_eager, y_eager, dout_eager

    def cal_torch_res(self, x, y, dout):
        x_t = x
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            y_t = y.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        out = x_t.expand_as(y)
        if self.dtype in int_types:
            return out, []
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, x, y, dout):
        x_t = x
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            y_t = paddle.cast(y, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.expand_as(x_t, y)
        if self.dtype in int_types:
            return out, []
        out_grads = paddle.grad([out], [x], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, y_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, y_eager, dout_eager)
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
        if self.dtype in int_types:
            np.testing.assert_array_equal(out_eager_np, self.out_torch)
        else:
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
                api="paddle.expand",
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
                api="paddle.expand",
            )


if __name__ == "__main__":
    unittest.main()
