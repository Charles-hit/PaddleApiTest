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

class TestReshapeInplaceDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, shape_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, shape_torch, dout_torch
        )
        del x_torch
        del shape_torch
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
        self.np_x = np.random.random(size=[4096, 1, 6144]).astype("float32") - 0.5
        self.np_shape = [0, 0, -1, 384]
        self.np_dout = np.random.random(size=[4096, 1, 16, 384]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
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
        shape_torch = self.np_shape
        for i in range(len(shape_torch)):
            if shape_torch[i] == 0:
                shape_torch[i] = self.np_x.shape[i]
        shape_torch = tuple(shape_torch)
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, shape_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        shape_eager = self.np_shape
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, shape_eager, dout_eager

    def cal_torch_res(self, x, shape, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        out = torch.reshape(x, shape)
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, shape, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        x_t = paddle.assign(x)
        out = paddle.reshape_(x_t, shape)
        out_grads = paddle.grad(
            [out], [x], grad_outputs=[dout]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, shape_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, shape_eager, dout_eager
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
            api="paddle.reshape_",
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
                api="paddle.reshape_",
            )

    def test_eager_stability(self):
        x_eager, shape_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, shape_eager, dout_eager
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
                x_eager, shape_eager, dout_eager
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
                api="paddle.reshape_",
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
                    api="paddle.reshape_",
                )


class TestReshapeInplaceDevelopCase1_FP16(TestReshapeInplaceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestReshapeInplaceDevelopCase1_BFP16(TestReshapeInplaceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase7_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 1, 64, 2]).astype("float32") - 0.5
        self.np_shape = [1, 8192, 1, 128]
        self.np_dout = np.random.random(size=[1, 8192, 1, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase7_FP16(TestReshapeInplaceDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase7_BFP16(TestReshapeInplaceDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase8_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 14, 128]).astype("float32") - 0.5
        self.np_shape = [1, 8192, 1792]
        self.np_dout = np.random.random(size=[1, 8192, 1792]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase8_FP16(TestReshapeInplaceDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase8_BFP16(TestReshapeInplaceDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase9_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 1]).astype("float32") - 0.5
        self.np_shape = [8192]
        self.np_dout = np.random.random(size=[8192]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase9_FP16(TestReshapeInplaceDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase9_BFP16(TestReshapeInplaceDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase10_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 5376]).astype("float32") - 0.5
        self.np_shape = [1, 8192, 14, 384]
        self.np_dout = np.random.random(size=[1, 8192, 14, 384]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase9_FP16(TestReshapeInplaceDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase9_BFP16(TestReshapeInplaceDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase11_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192]).astype("float32") - 0.5
        self.np_shape = [8192]
        self.np_dout = np.random.random(size=[8192]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase11_FP16(TestReshapeInplaceDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase11_BFP16(TestReshapeInplaceDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase12_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 31250]).astype("float32") - 0.5
        self.np_shape = [448000000]
        self.np_dout = np.random.random(size=[448000000]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase12_FP16(TestReshapeInplaceDevelopCase12_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase12_BFP16(TestReshapeInplaceDevelopCase12_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase13_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 5376]).astype("float32") - 0.5
        self.np_shape = [77070336]
        self.np_dout = np.random.random(size=[77070336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase13_FP16(TestReshapeInplaceDevelopCase13_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase13_BFP16(TestReshapeInplaceDevelopCase13_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase14_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 9632]).astype("float32") - 0.5
        self.np_shape = [138084352]
        self.np_dout = np.random.random(size=[138084352]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase14_FP16(TestReshapeInplaceDevelopCase14_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase14_BFP16(TestReshapeInplaceDevelopCase14_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestReshapeInplaceDevelopCase15_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336]).astype("float32") - 0.5
        self.np_shape = [14336]
        self.np_dout = np.random.random(size=[14336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase15_FP16(TestReshapeInplaceDevelopCase15_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase15_BFP16(TestReshapeInplaceDevelopCase15_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase16_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1792, 14336]).astype("float32") - 0.5
        self.np_shape = [25690112]
        self.np_dout = np.random.random(size=[25690112]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase16_FP16(TestReshapeInplaceDevelopCase16_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase16_BFP16(TestReshapeInplaceDevelopCase16_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase17_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[31250, 14336]).astype("float32") - 0.5
        self.np_shape = [448000000]
        self.np_dout = np.random.random(size=[448000000]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase17_FP16(TestReshapeInplaceDevelopCase17_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase17_BFP16(TestReshapeInplaceDevelopCase17_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase18_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[31250]).astype("float32") - 0.5
        self.np_shape = [31250]
        self.np_dout = np.random.random(size=[31250]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase18_FP16(TestReshapeInplaceDevelopCase18_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase18_BFP16(TestReshapeInplaceDevelopCase18_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase19_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4816, 14336]).astype("float32") - 0.5
        self.np_shape = [69042176]
        self.np_dout = np.random.random(size=[69042176]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase19_FP16(TestReshapeInplaceDevelopCase19_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase19_BFP16(TestReshapeInplaceDevelopCase19_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase20_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[5376]).astype("float32") - 0.5
        self.np_shape = [5376]
        self.np_dout = np.random.random(size=[5376]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase20_FP16(TestReshapeInplaceDevelopCase20_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase20_BFP16(TestReshapeInplaceDevelopCase20_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase21_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[8192, 128]).astype("float32") - 0.5
        self.np_shape = [1, 1, 8192, 128]
        self.np_dout = np.random.random(size=[1, 1, 8192, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase21_FP16(TestReshapeInplaceDevelopCase21_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase21_BFP16(TestReshapeInplaceDevelopCase21_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeInplaceDevelopCase22_FP32(TestReshapeInplaceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[9632]).astype("float32") - 0.5
        self.np_shape = [9632]
        self.np_dout = np.random.random(size=[9632]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeInplaceDevelopCase22_FP16(TestReshapeInplaceDevelopCase22_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeInplaceDevelopCase22_BFP16(TestReshapeInplaceDevelopCase22_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
