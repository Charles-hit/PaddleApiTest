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

class TestGaussianDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        out_torch = self.cal_torch_res()
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.size = [10944, 8192]
        self.mean = 0
        self.std = 0.00318928

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def cal_torch_res(self):
        out = torch.zeros(*self.size, device='cuda', dtype=convert_dtype_to_torch_type(self.dtype))
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.normal(mean=self.mean, std=self.std, size=self.size, out=out)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self):
        paddle.seed(0)
        out = paddle.tensor.random.gaussian(self.size, mean=self.mean, std=self.std, dtype=self.dtype)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def cal_static_res(self):
        out = paddle.tensor.random.gaussian(self.size, mean=self.mean, std=self.std, dtype=self.dtype)
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
            api="paddle.tensor.random.gaussian",
        )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                out_static = self.cal_static_res()
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            paddle.seed(0)
            paddle.framework.random._manual_program_seed(0)
            exe.run(sp)
            out = exe.run(
                mp,
                feed={},
                fetch_list=[out_static],
            )
            out_static = out[0]

        # compare develop static forward res with torch
        np_assert_accuracy(
            out_static,
            self.out_torch,
            self.atol,
            self.rtol,
            self.dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="static",
            fwd_or_bkd="forward",
            api="paddle.tensor.random.gaussian",
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
                api="paddle.tensor.random.gaussian",
            )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                out_static_pg = self.cal_static_res()
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            paddle.seed(0)
            paddle.framework.random._manual_program_seed(0)
            exe.run(sp)
            out = exe.run(
                mp,
                feed={},
                fetch_list=[out_static_pg]
            )
            out_static_baseline = out[0]
            for i in range(5):
                paddle.seed(0)
                paddle.framework.random._manual_program_seed(0)
                out = exe.run(
                    mp,
                    feed={},
                    fetch_list=[out_static_pg]
                )
                out_static = out[0]
                # test develop static forward stability
                np_assert_staility(
                    out_static,
                    out_static_baseline,
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="paddle.tensor.random.gaussian",
                )


class TestGaussianDevelopCase1_FP16(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [10944, 8192]
        self.mean = 0
        self.std = 0.00318928


class TestGaussianDevelopCase1_BFP16(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [10944, 8192]
        self.mean = 0
        self.std = 0.00318928


class TestGaussianDevelopCase2_FP32(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [2048, 4096]
        self.mean = 0
        self.std = 0.005


class TestGaussianDevelopCase2_FP16(TestGaussianDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [2048, 4096]
        self.mean = 0
        self.std = 0.005


class TestGaussianDevelopCase2_BFP16(TestGaussianDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [2048, 4096]
        self.mean = 0
        self.std = 0.005


class TestGaussianDevelopCase3_FP32(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [4096, 10944]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase3_FP16(TestGaussianDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4096, 10944]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase3_BFP16(TestGaussianDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [4096, 10944]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase4_FP32(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [4096, 6144]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase4_FP16(TestGaussianDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4096, 6144]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase4_BFP16(TestGaussianDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [4096, 6144]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase5_FP32(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [4096, 8192]
        self.mean = 0
        self.std = 0.00318928


class TestGaussianDevelopCase5_FP16(TestGaussianDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4096, 8192]
        self.mean = 0
        self.std = 0.00318928


class TestGaussianDevelopCase5_BFP16(TestGaussianDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [4096, 8192]
        self.mean = 0
        self.std = 0.00318928


class TestGaussianDevelopCase6_FP32(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [50176, 4096]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase6_FP16(TestGaussianDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [50176, 4096]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase6_BFP16(TestGaussianDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [50176, 4096]
        self.mean = 0
        self.std = 0.01


class TestGaussianDevelopCase7_FP32(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [50176, 8192]
        self.mean = 0
        self.std = 0.00637856


class TestGaussianDevelopCase7_FP16(TestGaussianDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [50176, 8192]
        self.mean = 0
        self.std = 0.00637856


class TestGaussianDevelopCase7_BFP16(TestGaussianDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [50176, 8192]
        self.mean = 0
        self.std = 0.00637856


class TestGaussianDevelopCase8_FP32(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [5472, 4096]
        self.mean = 0
        self.std = 0.005


class TestGaussianDevelopCase8_FP16(TestGaussianDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [5472, 4096]
        self.mean = 0
        self.std = 0.005


class TestGaussianDevelopCase8_BFP16(TestGaussianDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [5472, 4096]
        self.mean = 0
        self.std = 0.005


class TestGaussianDevelopCase9_FP32(TestGaussianDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [8192, 12288]
        self.mean = 0
        self.std = 0.00637856


class TestGaussianDevelopCase9_FP16(TestGaussianDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [8192, 12288]
        self.mean = 0
        self.std = 0.00637856


class TestGaussianDevelopCase9_BFP16(TestGaussianDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [8192, 12288]
        self.mean = 0
        self.std = 0.00637856


class TestGaussianDevelopCase10_FP32(TestGaussianDevelopCase1_FP32):
    def get_params(self):
        self.dtype = "float32"
        self.size = [8192, 21888]
        self.mean = 0
        self.std = 0.00637856


class TestGaussianDevelopCase10_FP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [8192, 21888]
        self.mean = 0
        self.std = 0.00637856


class TestGaussianDevelopCase10_BFP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size = [8192, 21888]
        self.mean = 0
        self.std = 0.00637856
class TestGaussianDevelopCase11_FP32(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [12528, 14336]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase11_FP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [12528, 14336]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase11_BFP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [12528, 14336]
        self.mean = 0
        self.std = 1



class TestGaussianDevelopCase12_FP32(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [14336, 5376]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase12_FP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336, 5376]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase12_BFP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336, 5376]
        self.mean = 0
        self.std = 1



class TestGaussianDevelopCase13_FP32(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [14336, 9632]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase13_FP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336, 9632]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase13_BFP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336, 9632]
        self.mean = 0
        self.std = 1



class TestGaussianDevelopCase14_FP32(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [1792, 14336]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase14_FP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1792, 14336]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase14_BFP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1792, 14336]
        self.mean = 0
        self.std = 1



class TestGaussianDevelopCase15_FP32(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [4816, 14336]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase15_FP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4816, 14336]
        self.mean = 0
        self.std = 1

class TestGaussianDevelopCase15_BFP16(TestGaussianDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4816, 14336]
        self.mean = 0
        self.std = 1



if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
