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

class TestEinsumDevelopCase1(unittest.TestCase):
    def init_dtype(self):
        self.dtype = self.in_0_dtype

    def init_params(self):
        # function_name: einsum, equation: i,j->ij, 
        # operands: in_0_shape: [18], 
        # in_0_dtype: paddle.float32,
        # in_1_shape: [16], 
        # in_1_dtype: paddle.float32,
        self.equation = 'i,j->ij'
        self.in_0_shape = [18]
        self.in_0_dtype = "float32"
        self.in_1_shape = [16]
        self.in_1_dtype = "float32"
        self.dout_shape = [18,16]
        self.init_dtype()

    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, y_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            self.equation, x_torch, y_torch, dout_torch
        )
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
        in_0_dtype_tmp = self.in_0_dtype if self.in_0_dtype != 'bfloat16' else "float32"
        in_1_dtype_tmp = self.in_1_dtype if self.in_1_dtype != 'bfloat16' else "float32"
        np_out_dtype_tmp = self.dtype if self.dtype != 'bfloat16' else "float32"
        self.np_in_0 = np.random.random(size=self.in_0_shape).astype(in_0_dtype_tmp) - 0.5
        self.np_in_1 = np.random.random(size=self.in_1_shape).astype(in_1_dtype_tmp) - 0.5
        self.np_dout = (
            np.random.random(size=self.dout_shape).astype(np_out_dtype_tmp) - 0.5
        )


    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def gen_torch_inputs_and_dout(self):
        in_0_torch = torch.tensor(
            self.np_in_0,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.in_0_dtype)
            if self.in_0_dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        in_1_torch = torch.tensor(
            self.np_in_1,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.in_1_dtype)
            if self.in_1_dtype != 'bfloat16'
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
        return in_0_torch, in_1_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        in_0_eager = paddle.to_tensor(
            self.np_in_0,
            dtype=self.in_0_dtype if self.in_0_dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        in_1_eager = paddle.to_tensor(
            self.np_in_1,
            dtype=self.in_1_dtype if self.in_1_dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        in_0_eager.stop_gradient = False
        in_1_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return in_0_eager, in_1_eager, dout_eager

    def cal_torch_res(self, equation, x, y, dout):
        x_t = x
        y_t = y
        dout_t = dout
        if self.in_0_dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
        if self.in_1_dtype == "bfloat16":
            y_t = y.to(dtype=torch.bfloat16)
        if self.dtype == "bfloat16":
            dout_t = dout.to(dtype=torch.bfloat16)
        out = torch.einsum(equation, x_t, y_t)
        out_grads = torch.autograd.grad([out], [x, y], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, equation, x, y, dout):
        x_t = x
        y_t = y
        dout_t = dout
        if self.in_0_dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
        if self.in_1_dtype == "bfloat16":
            y_t = paddle.cast(y, dtype="uint16")
        if self.dtype == "bfloat16":
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.einsum(equation, x_t, y_t)
        out_grads = paddle.grad([out], [x, y], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, y_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            self.equation, x_eager, y_eager, dout_eager
        )
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
            api="paddle.einsum",
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
                api="paddle.einsum",
            )

class TestEinsumDevelopCase2(TestEinsumDevelopCase1):
    def init_params(self):
        self.equation = 'i,j->ij'
        self.in_0_shape = [10]
        self.in_0_dtype = "float32"
        self.in_1_shape = [16]
        self.in_1_dtype = "float32"
        self.dout_shape = [10,16]
        self.init_dtype()

# class TestEinsumDevelopCase3(TestEinsumDevelopCase1):
#     def init_params(self):
#         self.equation = 'i,j->ij'
#         self.in_0_shape = [55]
#         self.in_0_dtype = "float32"
#         self.in_1_shape = [32]
#         self.in_1_dtype = "bfloat16"
#         self.dout_shape = [55,32]
#         self.init_dtype()

class TestEinsumDevelopCase3_1(TestEinsumDevelopCase1):
    def init_params(self):
        self.equation = 'i,j->ij'
        self.in_0_shape = [55]
        self.in_0_dtype = "float32"
        self.in_1_shape = [32]
        self.in_1_dtype = "float32"
        self.dout_shape = [55,32]
        self.init_dtype()

class TestEinsumDevelopCase3_2(TestEinsumDevelopCase1):
    def init_params(self):
        self.equation = 'i,j->ij'
        self.in_0_shape = [55]
        self.in_0_dtype = "bfloat16"
        self.in_1_shape = [32]
        self.in_1_dtype = "bfloat16"
        self.dout_shape = [55,32]
        self.init_dtype()

class TestEinsumDevelopCase4(TestEinsumDevelopCase1):
    def init_params(self):
        self.equation = 'i,j->ij'
        self.in_0_shape = [1]
        self.in_0_dtype = "float32"
        self.in_1_shape = [16]
        self.in_1_dtype = "float32"
        self.dout_shape = [1,16]
        self.init_dtype()

# 3d shape
class TestEinsumDevelopCase5(unittest.TestCase):
    def init_dtype(self):
        self.dtype = self.in_0_dtype
    
    def init_params(self):
        # td,hd,wd->thwd, 
        # operands: in_0_shape: [1, 96], 
        # in_0_dtype: paddle.bfloat16,
        # in_1_shape: [10, 96], 
        # in_1_dtype: paddle.bfloat16,
        # in_2_shape: [18, 96], 
        # in_2_dtype: paddle.bfloat16,
        self.equation = 'td,hd,wd->thwd'
        self.in_0_shape = [1, 96]
        self.in_0_dtype = "bfloat16"
        self.in_1_shape = [10, 96]
        self.in_1_dtype = "bfloat16"
        self.in_2_shape = [18, 96]
        self.in_2_dtype = "bfloat16"
        self.dout_shape = [1, 10, 18, 96]
        self.init_dtype()

    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, y_torch, z_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            self.equation, x_torch, y_torch, z_torch, dout_torch
        )
        del x_torch
        del y_torch
        del z_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_np_inputs_and_dout(self):
        in_0_dtype_tmp = self.in_0_dtype if self.in_0_dtype != 'bfloat16' else "float32"
        in_1_dtype_tmp = self.in_1_dtype if self.in_1_dtype != 'bfloat16' else "float32"
        in_2_dtype_tmp = self.in_2_dtype if self.in_2_dtype != 'bfloat16' else "float32"
        np_out_dtype_tmp = self.dtype if self.dtype != 'bfloat16' else "float32"
        self.np_in_0 = np.random.random(size=self.in_0_shape).astype(in_0_dtype_tmp) - 0.5
        self.np_in_1 = np.random.random(size=self.in_1_shape).astype(in_1_dtype_tmp) - 0.5
        self.np_in_2 = np.random.random(size=self.in_2_shape).astype(in_2_dtype_tmp) - 0.5
        self.np_dout = (
            np.random.random(size=self.dout_shape).astype(np_out_dtype_tmp) - 0.5
        )

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def gen_torch_inputs_and_dout(self):
        in_0_torch = torch.tensor(
            self.np_in_0,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.in_0_dtype)
            if self.in_0_dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        in_1_torch = torch.tensor(
            self.np_in_1,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.in_1_dtype)
            if self.in_1_dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        in_2_torch = torch.tensor(
            self.np_in_2,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.in_2_dtype)
            if self.in_2_dtype != 'bfloat16'
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
        return in_0_torch, in_1_torch, in_2_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        in_0_eager = paddle.to_tensor(
            self.np_in_0,
            dtype=self.in_0_dtype if self.in_2_dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        in_0_eager.stop_gradient = False
        in_1_eager = paddle.to_tensor(
            self.np_in_1,
            dtype=self.in_1_dtype if self.in_1_dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        in_1_eager.stop_gradient = False
        in_2_eager = paddle.to_tensor(
            self.np_in_2,
            dtype=self.in_2_dtype if self.in_2_dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        in_2_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return in_0_eager, in_1_eager, in_2_eager, dout_eager

    def cal_torch_res(self, equation, x, y, z, dout):
        x_t = x
        y_t = y
        z_t = z
        dout_t = dout
        if self.in_0_dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
        if self.in_1_dtype == "bfloat16":
            y_t = y.to(dtype=torch.bfloat16)
        if self.in_2_dtype == "bfloat16":
            z_t = z.to(dtype=torch.bfloat16)
        if self.dtype == "bfloat16":
            dout_t = dout.to(dtype=torch.bfloat16)
        out = torch.einsum(equation, x_t, y_t, z_t)
        out_grads = torch.autograd.grad([out], [x, y, z], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, equation, x, y, z, dout):
        x_t = x
        y_t = y
        z_t = z
        dout_t = dout
        if self.in_0_dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
        if self.in_1_dtype == "bfloat16":
            y_t = paddle.cast(y, dtype="uint16")
        if self.in_2_dtype == "bfloat16":
            z_t = paddle.cast(z, dtype="uint16")
        if self.dtype == "bfloat16":
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.einsum(equation, x_t, y_t, z_t)
        out_grads = paddle.grad([out], [x, y, z], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, y_eager, z_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            self.equation, x_eager, y_eager, z_eager, dout_eager
        )
        del x_eager
        del y_eager
        del z_eager
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
            api="paddle.einsum",
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
                api="paddle.einsum",
            )


if __name__ == '__main__':
    unittest.main()