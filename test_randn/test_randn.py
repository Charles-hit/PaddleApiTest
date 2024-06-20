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


class TestRandnDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        out_torch = self.cal_torch_res()
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.size = [12288, 64]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def cal_torch_res(self):
        out = torch.zeros(
            *self.size, device="cuda", dtype=convert_dtype_to_torch_type(self.dtype)
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.randn(size=self.size, out=out)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self):
        paddle.seed(0)
        out = paddle.randn(self.size, dtype=self.dtype)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        out_eager = self.cal_eager_res()
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
            api="paddle.randn",
        )

    def test_eager_stability(self):
        out_eager_baseline = self.cal_eager_res()
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager = self.cal_eager_res()
            out_eager = out_eager.numpy()
            # test develop eager forward stability
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.randn",
            )


class TestRandnDevelopCase2_BF16(TestRandnDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [12288, 9216]


class TestRandnDevelopCase3_BF16(TestRandnDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [1536, 12288]


class TestRandnDevelopCase4_BF16(TestRandnDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [32032, 12288]


class TestRandnDevelopCase5_BF16(TestRandnDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [12288, 32032]


class TestRandnDevelopCase6_BF16(TestRandnDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [4608, 12288]


class TestRandnDevelopCase7_BF16(TestRandnDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [12288, 1792]


if __name__ == "__main__":
    np.random.seed(2023)
    unittest.main()
