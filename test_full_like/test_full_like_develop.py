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

class TestFullLikeDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs()
        x_torch = self.gen_torch_inputs()
        out_torch = self.cal_torch_res(
            x_torch
        )
        del x_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.value = 0.0

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs(self):
        # init np array
        self.np_x = np.random.random(size=[1, 16, 4096, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def gen_torch_inputs(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=False,
        )
        return x_torch

    def gen_eager_inputs(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        return x_eager

    def gen_static_inputs(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        return x_static

    def cal_torch_res(self, x):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
        out = torch.full_like(x, self.value)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self, x):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
        out = paddle.full_like(x, self.value)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def cal_static_res(self, x):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
        out = paddle.full_like(x, self.value)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
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
            api="paddle.full_like",
        )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static = self.gen_static_inputs()
                out_static = self.cal_static_res(x_static)
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
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
            api="paddle.full_like",
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
                api="paddle.full_like",
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
                    api="paddle.full_like",
                )


class TestFullLikeDevelopCase1_FP16(TestFullLikeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = 0.0


class TestFullLikeDevelopCase1_BFP16(TestFullLikeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = 0.0


class TestFullLikeDevelopCase2_FP32(TestFullLikeDevelopCase1_FP32):
    def init_np_inputs(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 16, 4096, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def init_params(self):
        self.dtype = "float32"
        self.value = -np.inf


class TestFullLikeDevelopCase2_FP16(TestFullLikeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = -np.inf


class TestFullLikeDevelopCase2_BFP16(TestFullLikeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = -np.inf


class TestFullLikeDevelopCase3_FP32(TestFullLikeDevelopCase1_FP32):
    def init_np_inputs(self):
        # init np array 
        self.np_x = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def init_params(self):
        self.dtype = "float32"
        self.value = -np.inf


class TestFullLikeDevelopCase3_FP16(TestFullLikeDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = -np.inf


class TestFullLikeDevelopCase3_BFP16(TestFullLikeDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = -np.inf


class TestFullLikeDevelopCase4_FP32(TestFullLikeDevelopCase1_FP32):
    def init_np_inputs(self):
        # init np array 
        self.np_x = np.random.random(size=1).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def init_params(self):
        self.dtype = "float32"
        self.value = -np.inf


class TestFullLikeDevelopCase4_FP16(TestFullLikeDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.value = -np.inf


class TestFullLikeDevelopCase4_BFP16(TestFullLikeDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.value = -np.inf

# x_shape, fill_value, dtype
all_case = {
    # MP1
    (  (14) ,  1, 'None',  ), 
    (  (18) ,  1, 'None',  ), 
    (  (1) ,  0, 'None',  ), 
    (  (19) ,  1, 'None',  ), 
    (  (12) ,  1, 'None',  ), 
    (  (22) ,  1, 'None',  ), 
    (  (46) ,  1, 'None',  ), 
    (  (76) ,  1, 'None',  ), 
    (  (43) ,  1, 'None',  ), 
    (  (26) ,  1, 'None',  ), 
    (  (27) ,  1, 'None',  ), 
    (  (54) ,  1, 'None',  ), 
    (  (35) ,  1, 'None',  ), 
    (  (41) ,  1, 'None',  ), 
    (  (48) ,  1, 'None',  ), 
    (  (55) ,  1, 'None',  ), 
    (  (21) ,  1, 'None',  ), 
    (  (128, 1536) ,  0, 'None',  ), 
    (  (20) ,  1, 'None',  ), 
    (  (24) ,  1, 'None',  ), 
    (  (23) ,  1, 'None',  ), 
    (  (36) ,  1, 'None',  ), 
    (  (25) ,  1, 'None',  ), 
    (  (34) ,  1, 'None',  ), 
    (  (38) ,  1, 'None',  ), 
    (  (356) ,  1, 'None',  ), 
    (  (44) ,  1, 'None',  ), 
    (  (33) ,  1, 'None',  ), 
    (  (53) ,  1, 'None',  ), 
    (  (31) ,  1, 'None',  ), 
    (  (30) ,  1, 'None',  ), 
    (  (15) ,  1, 'None',  ), 
    (  (39) ,  1, 'None',  ), 
    (  (17) ,  1, 'None',  ), 
    (  (29) ,  1, 'None',  ), 
    (  (28) ,  1, 'None',  ), 
    (  (16) ,  1, 'None',  ), 
    (  (32) ,  1, 'None',  ), 
    (  (13) ,  1, 'None',  ), 
    (  (42) ,  1, 'None',  ), 
    (  (11) ,  1, 'None',  ), 
    (  (67) ,  1, 'None',  ), 
    # MP8
    (  (12) ,  1, 'None',  ), 
    (  (31) ,  1, 'None',  ), 
    (  (44) ,  1, 'None',  ), 
    (  (24) ,  1, 'None',  ), 
    (  (16) ,  1, 'None',  ), 
    (  (1) ,  0, 'None',  ), 
    (  (1264) ,  1, 'None',  ), 
    (  (47) ,  1, 'None',  ), 
    (  (55) ,  1, 'None',  ), 
    (  (19) ,  1, 'None',  ), 
    (  (26) ,  1, 'None',  ), 
    (  (14) ,  1, 'None',  ), 
    (  (40) ,  1, 'None',  ), 
    (  (18) ,  1, 'None',  ), 
    (  (30) ,  1, 'None',  ), 
    (  (25) ,  1, 'None',  ), 
    (  (15) ,  1, 'None',  ), 
    (  (23) ,  1, 'None',  ), 
    (  (36) ,  1, 'None',  ), 
    (  (1316) ,  1, 'None',  ), 
    (  (1354) ,  1, 'None',  ), 
    (  (49) ,  1, 'None',  ), 
    (  (1932) ,  1, 'None',  ), 
    (  (2018) ,  1, 'None',  ), 
    (  (28) ,  1, 'None',  ), 
    (  (29) ,  1, 'None',  ), 
    (  (1024, 1536) ,  0, 'None',  ), 
    (  (20) ,  1, 'None',  ), 
    (  (34) ,  1, 'None',  ), 
    (  (11) ,  1, 'None',  ), 
    (  (21) ,  1, 'None',  ), 
    (  (54) ,  1, 'None',  ), 
    (  (1960) ,  1, 'None',  ), 
    (  (58) ,  1, 'None',  ), 
    (  (39) ,  1, 'None',  ), 
    (  (35) ,  1, 'None',  ), 
    (  (1722) ,  1, 'None',  ), 
    (  (32) ,  1, 'None',  ), 
    (  (1890) ,  1, 'None',  ), 
    (  (43) ,  1, 'None',  ), 
    (  (37) ,  1, 'None',  ), 
    (  (48) ,  1, 'None',  ), 
    (  (33) ,  1, 'None',  ), 
    (  (17) ,  1, 'None',  ), 
    (  (38) ,  1, 'None',  ), 
    (  (27) ,  1, 'None',  ), 
    (  (13) ,  1, 'None',  ), 
    (  (22) ,  1, 'None',  ), 
}

class TestFullLikeDevelopCase5(unittest.TestCase):
    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array
        self.np_x = np.random.random(size=self.x_shape).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
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
        out = torch.full_like(x, fill_value=self.fill_value, dtype=None)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self, x):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
        out = paddle.full_like(x, self.fill_value, None)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        for (x_shape, fill_value, unuse_dtype) in all_case:
            for dtype in ('float16', 'float32', 'bfloat16'):
                self.x_shape = x_shape
                self.fill_value = fill_value
                self.dtype = dtype

                if not isinstance(self.x_shape, tuple):
                    self.x_shape = (self.x_shape, )

                self.init_threshold()
                # init np
                self.init_np_inputs_and_dout()
                # init torch
                x_torch= self.gen_torch_inputs_and_dout()
                out_torch = self.cal_torch_res(x_torch)
                del x_torch
                self.out_torch = out_torch.cpu().detach().numpy()
                del out_torch
                torch.cuda.empty_cache()
                # init paddle eager
                x_eager= self.gen_eager_inputs_and_dout()
                out_eager = self.cal_eager_res(x_eager)
                del x_eager
                paddle.device.cuda.empty_cache()
                out_eager_np = out_eager.numpy()
                del out_eager
                paddle.device.cuda.empty_cache()
                # compare develop eager forward res with torch
                np.testing.assert_allclose(
                    out_eager_np,
                    self.out_torch,
                    self.atol,
                    self.rtol,
                    err_msg=f'Develop: compare equal eager forward res with torch failed in x_shape: {self.x_shape}, fill_value: {self.fill_value}, x_dtype: {self.dtype},\n'
                )


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
