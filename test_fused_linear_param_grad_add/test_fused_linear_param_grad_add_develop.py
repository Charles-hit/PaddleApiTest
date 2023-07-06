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

from paddle import _C_ops

import paddle
from paddle.utils import map_structure

sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

def promote_dtype(x):
    if x.dtype in [torch.float16, torch.bfloat16]:
        return x.type(torch.float32)
    else:
        return x

def promote_dtype_paddle(x):
    if x.dtype in [paddle.float16, paddle.bfloat16]:
        return x.astype(paddle.float32)
    else:
        return x

def recreate(x, multi_precision):
    if isinstance(x, (list, tuple)):
        return [recreate(item, multi_precision) for item in x]

    if x is None:
        return None

    if multi_precision:
        x = promote_dtype(x)

    return torch.tensor(x.cpu().detach().numpy()).cuda()

def torch_fused_linear_param_grad_add(x, dy, dweight, dbias, multi_precision):
    x, dy, dweight, dbias = recreate([x, dy, dweight, dbias], multi_precision)
    dweight_tmp = torch.matmul(
        x.reshape([-1, x.shape[-1]]).transpose_(0, 1),
        dy.reshape([-1, dy.shape[-1]]),
    )
    if dweight is None:
        res_dweight = dweight_tmp
    else:
        assert dweight.shape == dweight_tmp.shape
        assert dweight.dtype == dweight_tmp.dtype
        res_dweight = dweight + dweight_tmp

    dbias_tmp = dy.reshape([-1, dy.shape[-1]]).sum(axis=0)
    if dbias is None:
        res_dbias = dbias_tmp
    else:
        assert dbias.shape == dbias_tmp.shape
        assert dbias.dtype == dbias_tmp.dtype
        res_dbias = dbias + dbias_tmp
    return promote_dtype(res_dweight), promote_dtype(res_dbias)

class TestFusedLinearParamGradAddDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.shape = [3, 4, 32]
        self.output_size = 128

        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        self.has_dweight = None
        self.has_dbias = None
        self.multi_precision = True
        x_torch, dy_torch, dweight_torch, dbias_torch = self.gen_torch_inputs_and_dout()
        out_dweight_torch, out_dbias_torch = self.cal_torch_res(
            x_torch, dy_torch, dweight_torch, dbias_torch
        )
        del x_torch
        del dy_torch
        del dweight_torch
        del dbias_torch
        self.out_dweight_torch = out_dweight_torch.cpu().numpy()
        self.out_dbias_torch = out_dbias_torch.cpu().numpy()

        del out_dweight_torch, out_dbias_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        dy_shape = self.shape[:-1] + [self.output_size]
        dweight_shape = [self.shape[-1], self.output_size]
        dbias_shape = [self.output_size]
        # init np array
        self.np_x = np.random.random(size=self.shape).astype("float32") - 0.5
        self.np_dy = np.random.random(size=dy_shape).astype("float32") - 0.5
        self.np_dweight = np.random.random(size=dweight_shape).astype("float32") - 0.5
        self.np_dbias = np.random.random(size=dbias_shape).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = np.random.random(size=self.shape).astype("float32")
            self.np_dy = np.random.random(size=dy_shape).astype("float32")
            self.np_dweight = np.random.random(size=dweight_shape).astype("float32")
            self.np_dbias = np.random.random(size=dbias_shape).astype("float32")


    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
        )
        dy_torch = torch.tensor(
            self.np_dy,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
        )
        dweight_torch = torch.tensor(
            self.np_dweight,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
        )
        dbias_torch = torch.tensor(
            self.np_dbias,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
        )
        return x_torch, dy_torch, dweight_torch, dbias_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dy_eager = paddle.to_tensor(
            self.np_dy,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dy_eager.stop_gradient = False
        dweight_eager = paddle.to_tensor(
            self.np_dweight,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dweight_eager.stop_gradient = False
        dbias_eager = paddle.to_tensor(
            self.np_dbias,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dbias_eager.stop_gradient = False
        return x_eager, dy_eager, dweight_eager, dbias_eager

    def cal_torch_res(self, x, dy, dweight, dbias):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dy = dy.to(dtype=torch.bfloat16)
            dweight = dweight.to(dtype=torch.bfloat16)
            dbias = dbias.to(dtype=torch.bfloat16)
        out_dweight, out_dbias = torch_fused_linear_param_grad_add(x, dy, dweight, dbias, self.multi_precision)

        if self.dtype == "bfloat16":
            out_dweight = out_dweight.to(dtype=torch.float32)
            out_dbias = out_dbias.to(dtype=torch.float32)
        return out_dweight, out_dbias

    def cal_eager_res(self, x, dy, dweight, dbias):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dy = paddle.cast(dy, dtype="uint16")
            dweight = paddle.cast(dweight, dtype="uint16")
            dbias = paddle.cast(dbias, dtype="uint16")
        if self.multi_precision:
            dweight = promote_dtype_paddle(dweight)
            dbias = promote_dtype_paddle(dbias)
        out_dweight, out_dbias = _C_ops.fused_linear_param_grad_add(
            x, dy, dweight, dbias, self.multi_precision
            )

        return out_dweight, out_dbias

    def test_eager_accuracy(self):
        x_eager, dy_eager, dweight_eager, dbias_eager = self.gen_eager_inputs_and_dout()
        out_dweight_eager, out_dbias_eager = self.cal_eager_res(
            x_eager, dy_eager, dweight_eager, dbias_eager
        )
        del x_eager
        del dy_eager
        del dweight_eager
        del dbias_eager
        paddle.device.cuda.empty_cache()
        out_dweight_eager_np = out_dweight_eager.numpy()
        out_dbias_eager_np = out_dbias_eager.numpy()

        del out_dweight_eager
        del out_dbias_eager
        paddle.device.cuda.empty_cache()
        # compare develop eager forward res with torch
        np_assert_accuracy(
            out_dweight_eager_np,
            self.out_dweight_torch,
            self.atol,
            self.rtol,
            self.dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="eager",
            fwd_or_bkd="forward",
            api="run_fused_linear_param_grad_add",
        )
        np_assert_accuracy(
            out_dbias_eager_np,
            self.out_dbias_torch,
            self.atol,
            self.rtol,
            self.dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="eager",
            fwd_or_bkd="forward",
            api="run_fused_linear_param_grad_add",
        )

    def test_eager_stability(self):
        x_eager, dy_eager, dweight_eager, dbias_eager = self.gen_eager_inputs_and_dout()
        out_dweight_eager_baseline, out_dbias_eager_baseline = self.cal_eager_res(
            x_eager, dy_eager, dweight_eager, dbias_eager
        )
        out_dweight_eager_baseline_np = out_dweight_eager_baseline.numpy()
        out_dbias_eager_baseline_np = out_dbias_eager_baseline.numpy()

        del out_dweight_eager_baseline
        del out_dbias_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            x_eager, dy_eager, dweight_eager, dbias_eager = self.gen_eager_inputs_and_dout()
            out_dweight_eager, out_dbias_eager = self.cal_eager_res(
                x_eager, dy_eager, dweight_eager, dbias_eager
            )
            out_dweight_eager = out_dweight_eager.numpy()
            out_dbias_eager = out_dbias_eager.numpy()

            # test develop eager forward stability
            np_assert_staility(
                out_dweight_eager,
                out_dweight_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="run_fused_linear_param_grad_add",
            )
            np_assert_staility(
                out_dbias_eager,
                out_dbias_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="run_fused_linear_param_grad_add",
            )

class TestFusedLinearParamGradAddDevelopCase1_FP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestFusedLinearParamGradAddDevelopCase1_BFP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestFusedLinearParamGradAddDevelopCase2(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.shape = [1, 16, 4096, 128]
        self.dtype = "float32"


class TestFusedLinearParamGradAddDevelopCase2_FP16(TestFusedLinearParamGradAddDevelopCase2):
    def init_params(self):
        self.dtype = "float16"


class TestFusedLinearParamGradAddDevelopCase2_BFP16(TestFusedLinearParamGradAddDevelopCase2):
    def init_params(self):
        self.dtype = "bfloat16"

class TestFusedLinearParamGradAddDevelopCase3_FP32(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape = [1, 8192, 14336]
        self.output_size = 5376 

class TestFusedLinearParamGradAddDevelopCase3_FP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape = [1, 8192, 14336]
        self.output_size = 5376 

class TestFusedLinearParamGradAddDevelopCase3_BFP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape = [1, 8192, 14336]
        self.output_size = 5376 


class TestFusedLinearParamGradAddDevelopCase4_FP32(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape = [1, 8192, 14336]
        self.output_size = 9632

class TestFusedLinearParamGradAddDevelopCase4_FP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape = [1, 8192, 14336]
        self.output_size = 9632

class TestFusedLinearParamGradAddDevelopCase4_BFP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape = [1, 8192, 14336]
        self.output_size = 9632

class TestFusedLinearParamGradAddDevelopCase5_FP32(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape = [1, 8192, 1792]
        self.output_size = 14336

class TestFusedLinearParamGradAddDevelopCase5_FP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape = [1, 8192, 1792]
        self.output_size = 14336

class TestFusedLinearParamGradAddDevelopCase5_BFP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape = [1, 8192, 1792]
        self.output_size = 14336

class TestFusedLinearParamGradAddDevelopCase6_FP32(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape = [1, 8192, 4816]
        self.output_size = 14336

class TestFusedLinearParamGradAddDevelopCase6_FP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape = [1, 8192, 4816]
        self.output_size = 14336

class TestFusedLinearParamGradAddDevelopCase6_BFP16(TestFusedLinearParamGradAddDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape = [1, 8192, 4816]
        self.output_size = 14336


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
