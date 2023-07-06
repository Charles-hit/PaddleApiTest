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

class TestFill_DevelopCase1_FP32(unittest.TestCase):
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
        self.value = 0

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10944]).astype("float32") - 0.5
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
        x.fill_(self.value)
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.float32)
        return x

    def cal_eager_res(self, x):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
        x.fill_(self.value)
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
            api="paddle.fill_",
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
                api="paddle.fill_",
            )


class TestFill_DevelopCase1_FP16(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0


class TestFill_DevelopCase1_BFP16(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0


class TestFill_DevelopCase2_FP32(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[2048, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestFill_DevelopCase2_FP16(TestFill_DevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0


class TestFill_DevelopCase2_BFP16(TestFill_DevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0


class TestFill_DevelopCase3_FP32(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestFill_DevelopCase3_FP16(TestFill_DevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0


class TestFill_DevelopCase3_BFP16(TestFill_DevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0


class TestFill_DevelopCase4_FP32(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 10944]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestFill_DevelopCase4_FP16(TestFill_DevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0


class TestFill_DevelopCase4_BFP16(TestFill_DevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0


class TestFill_DevelopCase5_FP32(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 6144]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestFill_DevelopCase5_FP16(TestFill_DevelopCase5_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0


class TestFill_DevelopCase5_BFP16(TestFill_DevelopCase5_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0


class TestFill_DevelopCase6_FP32(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[50176, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestFill_DevelopCase6_FP16(TestFill_DevelopCase6_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0


class TestFill_DevelopCase6_BFP16(TestFill_DevelopCase6_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0


class TestFill_DevelopCase7_FP32(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[5472, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestFill_DevelopCase7_FP16(TestFill_DevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0


class TestFill_DevelopCase7_BFP16(TestFill_DevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0


class TestFill_DevelopCase8_FP32(TestFill_DevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0
        self.size = [6144]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=self.size).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestFill_DevelopCase8_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [6144]


class TestFill_DevelopCase8_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0
        self.size = [6144]

class TestFill_DevelopCase9_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0
        self.size = [1]

class TestFill_DevelopCase9_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [1]

class TestFill_DevelopCase9_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [1]


class TestFill_DevelopCase10_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [12528, 14336]
        self.value = 0

class TestFill_DevelopCase10_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [12528, 14336]

class TestFill_DevelopCase10_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [12528, 14336]
        self.value = 0



class TestFill_DevelopCase11_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [14336]
        self.value = 0

class TestFill_DevelopCase11_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [14336]

class TestFill_DevelopCase11_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [14336]



class TestFill_DevelopCase12_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0
        self.size = [14336, 5376]

class TestFill_DevelopCase12_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [14336, 5376]

class TestFill_DevelopCase12_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [14336, 5376]



class TestFill_DevelopCase13_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0
        self.size = [14336, 9632]

class TestFill_DevelopCase13_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [14336, 9632]

class TestFill_DevelopCase13_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [14336, 9632]



class TestFill_DevelopCase14_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [1792, 14336]
        self.value = 0

class TestFill_DevelopCase14_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [1792, 14336]

class TestFill_DevelopCase14_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [1792, 14336]



class TestFill_DevelopCase15_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.value = 0
        self.dtype = "float32"
        self.size = [4816, 14336]

class TestFill_DevelopCase15_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4816, 14336]
        self.value = 0

class TestFill_DevelopCase15_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [4816, 14336]



class TestFill_DevelopCase16_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0
        self.size = [5376]

class TestFill_DevelopCase16_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [5376]
        self.value = 0

class TestFill_DevelopCase16_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [5376]


class TestFill_DevelopCase17_FP32(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.value = 0
        self.size = [9632]

class TestFill_DevelopCase17_FP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [9632]

class TestFill_DevelopCase17_BFP16(TestFill_DevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0
        self.size = [9632]


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
