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

class TestEmptyDevelopCase1_FP32(unittest.TestCase):
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

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array
        self.np_x = np.random.random(size=[2048 , 1 , 4096]).astype("float32") - 0.5
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
        x = torch.empty(x.shape,  dtype=x.dtype , device = 'cuda')
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.float32)
        return x

    def cal_eager_res(self, x):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
        x=paddle.empty(shape=x.shape,  dtype=x.dtype)
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
            api="paddle.empty",
        )

    def test_eager_stability(self):
        x_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline = self.cal_eager_res(x_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager = self.cal_eager_res(x_eager)
            out_eager = out_eager.numpy()
            # test develop eager forward stability
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.empty",
            )


class TestEmptyDevelopCase1_FP16(TestEmptyDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestEmptyDevelopCase1_BFP16(TestEmptyDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestEmptyDevelopCase2_FP32(TestEmptyDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"

    def init_np_inputs_and_dout(self):
        # init np array
        self.np_x = np.random.random(size=[4096 ,1 ,4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestEmptyDevelopCase2_FP16(TestEmptyDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestEmptyDevelopCase2_BFP16(TestEmptyDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestEmptyDevelopCase3_FP32(TestEmptyDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"

    def init_np_inputs_and_dout(self):
        # init np array
        self.np_x = np.random.random(size=[1, 8192, 14336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestEmptyDevelopCase3_FP16(TestEmptyDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestEmptyDevelopCase3_BFP16(TestEmptyDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestEmptyDevelopCase4_FP32(TestEmptyDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"

    def init_np_inputs_and_dout(self):
        # init np array
        self.np_x = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestEmptyDevelopCase4_FP16(TestEmptyDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestEmptyDevelopCase4_BFP16(TestEmptyDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
