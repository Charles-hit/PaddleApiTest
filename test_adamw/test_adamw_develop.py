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

from curses import flash
from symbol import parameters
import sys
from typing_extensions import Self
import unittest
import gc

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

class TestAdamWDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.index = 0
        self.use_weight_deacy = np.random.randint(0 , 1, size=[16])
        self.gen_torch_data()

    def gen_torch_data(self):
        self.init_np_inputs_and_grad()
        x_torch, grad_torch = self.gen_torch_inputs_and_grad()
        out_torch = self.cal_torch_res(
            x_torch, grad_torch
        )
        del x_torch
        del grad_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.shapes = [4096]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_grad(self):
        # init np array 
        self.np_x = np.random.random(size=self.shapes).astype("float32") - 0.5
        self.np_grad = np.random.random(size=self.shapes).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_grad = self.np_grad.astype("float16")

    def gen_torch_inputs_and_grad(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        grad_torch = torch.tensor(
            self.np_grad,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, grad_torch

    def gen_eager_inputs_and_grad(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        grad_eager = paddle.to_tensor(
            self.np_grad,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        grad_eager.stop_gradient = False
        return x_eager, grad_eager

    def gen_static_inputs_and_grad(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        grad_static = paddle.static.data(
            'grad',
            shape=self.np_grad.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        grad_static.stop_gradient = False
        return x_static, grad_static

    def cal_torch_res(self, x, grad):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            grad = grad.to(dtype=torch.bfloat16)

        x.retain_grad()

        opt = torch.optim.AdamW([x],
                                lr=0.001,
                                betas=(0.9, 0.95),
                                eps=1e-08,
                                weight_decay= 0.1 if self.use_weight_deacy[self.index] else 0.0,
                                )

        x.grad = grad
        
        for _ in range(5):
            opt.step()

        del opt
        torch.cuda.empty_cache()

        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.float32)

        
        return x

    def cal_eager_res(self, x, grad):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            grad = paddle.cast(grad, dtype="uint16")
        
        x = x.detach()
        
        x.stop_gradient = False
        
        opt = paddle.optimizer.AdamW(learning_rate=0.001,
                                     beta1=0.9,
                                     beta2=0.95,
                                     epsilon=1e-08,
                                     parameters=[x],
                                     weight_decay= 0.1 if self.use_weight_deacy[self.index] else 0.0,
                                     multi_precision=True,
                                    )

        x.grad = grad

        for _ in range(5):
            opt.step()

        del opt
        gc.collect()
        paddle.device.cuda.empty_cache()
        
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="float32")
        return x

    def cal_static_res(self, x, grad):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            grad = paddle.cast(grad, dtype="uint16")

        x.regularizer = None
        x.stop_gradient = False

        opt = paddle.optimizer.AdamW(learning_rate=0.001,
                                     beta1=0.9,
                                     beta2=0.95,
                                     epsilon=1e-08,
                                     parameters=x,
                                     weight_decay= 0.1 if self.use_weight_deacy[self.index] else 0.0,
                                     multi_precision=True,
                                    )

        x.grad = grad
        
        for _ in range(5):
            opt._apply_optimize(loss=x, startup_program=paddle.fluid.default_main_program(), params_grads=[(x, x.grad)])

        del opt
        paddle.device.cuda.empty_cache()

        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="float32")
        return x

    def test_eager_accuracy(self):

        x_eager, grad_eager = self.gen_eager_inputs_and_grad()
        out_eager = self.cal_eager_res(
            x_eager, grad_eager
        )
        del x_eager
        del grad_eager
        out_eager_np = out_eager.numpy()
        del out_eager
        gc.collect()
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
            api="paddle.optimizer.adamw",
        )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    grad_static,
                ) = self.gen_static_inputs_and_grad()
                out_static = self.cal_static_res(
                    x_static,
                    grad_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "grad": self.np_grad},
                fetch_list=[out_static],
            )
            out_static= out[0]
        
            del exe
            gc.collect()
            paddle.device.cuda.empty_cache()

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
            api="paddle.optimizer.adamw",
        )

    def test_eager_stability(self):
        x_eager, grad_eager = self.gen_eager_inputs_and_grad()
        out_eager_baseline = self.cal_eager_res(
            x_eager, grad_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        del x_eager
        del grad_eager
        paddle.device.cuda.empty_cache()

        for i in range(10):
            x_eager, grad_eager = self.gen_eager_inputs_and_grad()
            out_eager = self.cal_eager_res(
                x_eager, grad_eager
            )
            out_eager_np = out_eager.numpy()
            # test develop eager forward stability
            np_assert_staility(
                out_eager_np,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.optimizer.adamw",
            )

            del x_eager
            del grad_eager
            del out_eager
            gc.collect()
            paddle.device.cuda.empty_cache()

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    grad_static,
                ) = self.gen_static_inputs_and_grad()

                out_static_pg = self.cal_static_res(
                    x_static,
                    grad_static,
                )

            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "grad": self.np_grad},
                fetch_list=[out_static_pg],
            )
            out_static_baseline = out[0]

            del exe
            gc.collect()

            for i in range(10):
                exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
                exe.run(sp)
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "grad": self.np_grad},
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
                    api="paddle.optimizer.adamw",
                )

                del exe
                gc.collect()
                paddle.device.cuda.empty_cache()

test_shape = [[1], [4096], [50176, 8192], [2048, 4096], [4096, 10944],
                    [10944], [5472, 4096], [50176, 4096], [6144], 
                    [8192, 12288], [12288], [4096, 8192], [4096, 6144],
                    [8192, 21888], [21888], [10944, 8192], [8192],
                    [12528, 14336], [5376], [1792, 14336], [14336], [14336, 9632], [9632], [4816, 14336],  [14336, 5376]]
class TestAdamWDevelopCase1_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[1]

class TestAdamWDevelopCase1_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[1]

class TestAdamWDevelopCase2_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[2]

class TestAdamWDevelopCase2_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[2]

class TestAdamWDevelopCase2_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[2]

class TestAdamWDevelopCase3_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[3]

class TestAdamWDevelopCase3_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[3]

class TestAdamWDevelopCase3_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[3]

class TestAdamWDevelopCase4_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[4]

class TestAdamWDevelopCase4_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[4]

class TestAdamWDevelopCase4_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[4]

class TestAdamWDevelopCase5_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[5]

class TestAdamWDevelopCase5_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[5]

class TestAdamWDevelopCase5_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[5]

class TestAdamWDevelopCase6_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[6]

class TestAdamWDevelopCase6_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[6]

class TestAdamWDevelopCase6_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[6]

class TestAdamWDevelopCase7_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[7]

class TestAdamWDevelopCase7_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[7]

class TestAdamWDevelopCase7_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[7]

class TestAdamWDevelopCase8_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[8]

class TestAdamWDevelopCase8_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[8]

class TestAdamWDevelopCase8_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[8]

class TestAdamWDevelopCase9_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[9]

class TestAdamWDevelopCase9_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[9]

class TestAdamWDevelopCase9_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[9]

class TestAdamWDevelopCase10_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[10]

class TestAdamWDevelopCase10_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[10]

class TestAdamWDevelopCase10_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[10]

class TestAdamWDevelopCase11_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[11]

class TestAdamWDevelopCase11_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[11]

class TestAdamWDevelopCase11_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[11]

class TestAdamWDevelopCase12_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[12]

class TestAdamWDevelopCase12_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[12]

class TestAdamWDevelopCase12_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[12]

class TestAdamWDevelopCase13_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[13]

class TestAdamWDevelopCase13_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[13]

class TestAdamWDevelopCase13_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[13]

class TestAdamWDevelopCase14_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[14]

class TestAdamWDevelopCase14_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[14]

class TestAdamWDevelopCase14_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[14]

class TestAdamWDevelopCase15_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[15]

class TestAdamWDevelopCase15_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[15]

class TestAdamWDevelopCase15_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[15]

class TestAdamWDevelopCase16_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[16]

class TestAdamWDevelopCase16_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[16]

class TestAdamWDevelopCase16_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[16]

class TestAdamWDevelopCase17_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[17]

class TestAdamWDevelopCase17_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[17]

class TestAdamWDevelopCase17_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[17]

class TestAdamWDevelopCase18_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[18]

class TestAdamWDevelopCase18_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[18]

class TestAdamWDevelopCase18_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[18]

class TestAdamWDevelopCase19_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[19]

class TestAdamWDevelopCase19_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[19]

class TestAdamWDevelopCase19_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[19]

class TestAdamWDevelopCase20_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[20]

class TestAdamWDevelopCase20_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[20]

class TestAdamWDevelopCase20_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[20]

class TestAdamWDevelopCase21_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[21]

class TestAdamWDevelopCase21_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[21]

class TestAdamWDevelopCase21_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[21]

class TestAdamWDevelopCase22_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[22]

class TestAdamWDevelopCase22_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[22]

class TestAdamWDevelopCase22_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[22]

class TestAdamWDevelopCase23_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[23]

class TestAdamWDevelopCase23_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[23]

class TestAdamWDevelopCase23_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[23]

class TestAdamWDevelopCase24_FP32(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shapes = test_shape[24]

class TestAdamWDevelopCase24_FP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shapes = test_shape[24]

class TestAdamWDevelopCase24_BFP16(TestAdamWDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shapes = test_shape[24]
    
if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
