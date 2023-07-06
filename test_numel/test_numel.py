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

class TestNumelDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs()
        x_torch = self.gen_torch_inputs()
        out_torch = self.cal_torch_res(
            x_torch
        )
        del x_torch
        out_torch = torch.tensor([out_torch])
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()
    
    def init_params(self):
        self.dtype = "float32"
    
    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs(self):
        # init np array 
        self.np_x = np.random.random(size=[14336]).astype("float32")
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def gen_torch_inputs(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype),
            requires_grad=False,
        )
        return x_torch

    def gen_eager_inputs(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = True
        return x_eager

    def gen_static_inputs(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
        )
        x_static.stop_gradient = True
        return x_static

    def cal_torch_res(self, x):
        out = torch.numel(x)
        return out

    def cal_eager_res(self, x):
        out = paddle.numel(x)
        return out

    def cal_static_res(self, x):
        out = paddle.numel(x)
        return out

    def test_eager_accuracy(self):
        x_eager = self.gen_eager_inputs()
        out_eager = self.cal_eager_res(
            x_eager
        )
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
            api="paddle.numel",
        )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static = self.gen_static_inputs()
                (out_static) = self.cal_static_res(
                    x_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            print("out_static.dtype:", out_static.dtype)
            print("self.np_x.dtype:", self.np_x.dtype)
            out = exe.run(
                mp,
                feed={"x": self.np_x},
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
            api="paddle.numel",
        )

    def test_eager_stability(self):
        x_eager = self.gen_eager_inputs()
        out_eager_baseline = self.cal_eager_res(
            x_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager = self.cal_eager_res(
                x_eager
            )
            out_eager = out_eager.numpy()
            # test develop eager forward stability
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.numel",
            )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static = self.gen_static_inputs()
                out_static_pg = self.cal_static_res(x_static)
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x},
                fetch_list=[out_static_pg],
            )
            out_static_baseline = out[0]
            for i in range(5):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x},
                    fetch_list=[out_static_pg],
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
                    api="paddle.numel",
                )
class TestNumelDevelopCase1_FP16(TestNumelDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase1_BFP16(TestNumelDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestNumelDevelopCase2_FP32(TestNumelDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 31250]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

class TestNumelDevelopCase2_FP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase2_BFP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestNumelDevelopCase3_FP32(TestNumelDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 5376]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

class TestNumelDevelopCase3_FP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase3_BFP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestNumelDevelopCase4_FP32(TestNumelDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 9632]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

class TestNumelDevelopCase4_FP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase4_BFP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestNumelDevelopCase5_FP32(TestNumelDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1792, 14336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

class TestNumelDevelopCase5_FP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase5_BFP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestNumelDevelopCase6_FP32(TestNumelDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[31250]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

class TestNumelDevelopCase6_FP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase6_BFP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestNumelDevelopCase7_FP32(TestNumelDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4816, 14336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

class TestNumelDevelopCase7_FP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase7_BFP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
    
class TestNumelDevelopCase8_FP32(TestNumelDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[5376]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

class TestNumelDevelopCase8_FP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase8_BFP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestNumelDevelopCase9_FP32(TestNumelDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[9632]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

class TestNumelDevelopCase9_FP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestNumelDevelopCase9_BFP16(TestNumelDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
