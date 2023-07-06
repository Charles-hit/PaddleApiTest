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

class TestNotEqualDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and()
        x_torch, y_torch = self.gen_torch_inputs()
        out_torch = self.cal_torch_res(
            x_torch, y_torch
        )
        del x_torch, y_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192]).astype("float32") - 0.5
        self.np_y = np.random.random(size=[1, 8192]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_y = self.np_y.astype("float16")

    def gen_torch_inputs(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=False,
        )
        y_torch = torch.tensor(
            self.np_y,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=False,
        )
        return x_torch, y_torch

    def gen_eager_inputs(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = True
        y_eager = paddle.to_tensor(
            self.np_y,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        y_eager.stop_gradient = True
        return x_eager, y_eager

    def gen_static_inputs(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = True
        y_static = paddle.static.data(
            'y',
            shape=self.np_y.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        y_static.stop_gradient = True
        return x_static, y_static

    def cal_torch_res(self, x, y):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            y = y.to(dtype=torch.bfloat16)
        out = torch.not_equal(x, y)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self, x, y):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            y = paddle.cast(y, dtype="uint16")
        out = paddle.not_equal(x, y)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def cal_static_res(self, x, y):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            y = paddle.cast(y, dtype="uint16")
        out = paddle.not_equal(x, y)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        x_eager, y_eager = self.gen_eager_inputs()
        out_eager = self.cal_eager_res(
            x_eager, y_eager
        )
        del x_eager, y_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        del out_eager
        paddle.device.cuda.empty_cache()
        # compare develop eager forward res with torch
        np.testing.assert_array_equal(out_eager_np, self.out_torch)
    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    y_static,
                ) = self.gen_static_inputs()
                (out_static) = self.cal_static_res(
                    x_static,
                    y_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out_static],
            )
            out_static = out[0]

        # compare develop static forward res with torch
        np.testing.assert_array_equal(out_static, self.out_torch)

    def test_eager_stability(self):
        x_eager, y_eager= self.gen_eager_inputs()
        out_eager_baseline = self.cal_eager_res(
            x_eager, y_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager = self.cal_eager_res(
                x_eager, y_eager
            )
            out_eager = out_eager.numpy()
            # test develop eager forward stability
            np.testing.assert_allclose(out_eager, out_eager_baseline_np)

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    y_static,
                ) = self.gen_static_inputs()
                (out_static_pg) = self.cal_static_res(
                    x_static,
                    y_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out_static_pg],
            )
            out_static_baseline = out[0]
            for i in range(5):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "y": self.np_y},
                    fetch_list=[out_static_pg],
                )
                out_static = out[0]
                # test develop static forward stability
                np.testing.assert_allclose(out_static, out_static_baseline)


class TestNotEqualDevelopCase1_FP16(TestNotEqualDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNotEqualDevelopCase1_BFP16(TestNotEqualDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        
if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()