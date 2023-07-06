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

class TestAddInplaceDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, y_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, y_torch, dout_torch
        )
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

    def init_params(self):
        self.dtype = "float32"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096]).astype("float32") - 0.5
        self.np_y = np.random.random(size=[4096]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_y = self.np_y.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def gen_torch_inputs_and_dout(self):
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
        return x_torch, y_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
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
        return x_eager, y_eager, dout_eager

    def cal_torch_res(self, x, y, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            y = y.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        x_t = x.clone()
        y_t = y.clone()
        out = torch.Tensor.add_(x_t, y_t)
        out_grads = torch.autograd.grad([out], [x, y], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, y, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            y = paddle.cast(y, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        x_t = paddle.assign(x)
        y_t = paddle.assign(y)
        out = paddle.tensor.add_(x_t, y_t)
        out_grads = paddle.grad([out], [x, y], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, y_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, y_eager, dout_eager
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
            api="paddle.tensor.add_",
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
                api="paddle.tensor.add_",
            )

    def test_eager_stability(self):
        x_eager, y_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, y_eager, dout_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager_baseline,
        )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager, out_grads_eager = self.cal_eager_res(
                x_eager, y_eager, dout_eager
            )
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                lambda x: x.numpy(),
                out_grads_eager,
            )
            # test develop eager forward stability
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.tensor.add_",
            )
            # test develop eager backward stability
            for idx in range(len(out_grads_eager)):
                np_assert_staility(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="paddle.tensor.add_",
                )

class TestAddInplaceDevelopCase1_FP16(TestAddInplaceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestAddInplaceDevelopCase1_BFP16(TestAddInplaceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestAddInplaceDevelopCase2_FP32(TestAddInplaceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size =[50176, 4096]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=self.size).astype("float32") - 0.5
        self.np_y = np.random.random(size=self.size).astype("float32") - 0.5
        self.np_dout = np.random.random(size=self.size).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_y = self.np_y.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestAddInplaceDevelopCase2_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size =[50176, 4096]

class TestAddInplaceDevelopCase2_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.size =[50176, 4096]
class TestAddInplaceDevelopCase3_FP32(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [12528, 14336]

class TestAddInplaceDevelopCase3_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [12528, 14336]

class TestAddInplaceDevelopCase3_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [12528, 14336]



class TestAddInplaceDevelopCase4_FP32(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [14336, 5376]

class TestAddInplaceDevelopCase4_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336, 5376]

class TestAddInplaceDevelopCase4_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336, 5376]



class TestAddInplaceDevelopCase5_FP32(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [14336, 9632]

class TestAddInplaceDevelopCase5_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336, 9632]

class TestAddInplaceDevelopCase5_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336, 9632]



class TestAddInplaceDevelopCase6_FP32(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [14336]

class TestAddInplaceDevelopCase6_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336]

class TestAddInplaceDevelopCase6_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [14336]



class TestAddInplaceDevelopCase7_FP32(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [1792, 14336]

class TestAddInplaceDevelopCase7_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1792, 14336]

class TestAddInplaceDevelopCase7_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [1792, 14336]



class TestAddInplaceDevelopCase8_FP32(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [4816, 14336]

class TestAddInplaceDevelopCase8_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4816, 14336]

class TestAddInplaceDevelopCase8_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [4816, 14336]



class TestAddInplaceDevelopCase9_FP32(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [5376]

class TestAddInplaceDevelopCase9_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [5376]

class TestAddInplaceDevelopCase9_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [5376]



class TestAddInplaceDevelopCase10_FP32(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [9632]

class TestAddInplaceDevelopCase10_FP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [9632]

class TestAddInplaceDevelopCase10_BFP16(TestAddInplaceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [9632]

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
