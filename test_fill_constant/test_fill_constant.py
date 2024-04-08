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

import paddle
import torch

sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)


def generate_np_inputs():
    shape_case1 = [1]
    fill_value_case1 = 0.0
    dtype_case1 = 'float32'

    shape_case2 = [1]
    fill_value_case2 = 1.0
    dtype_case2 = 'float32'

    np.savez(
        "./inputs_case1.npz",
        shape=shape_case1,
        fill_value=fill_value_case1,
        dtype=dtype_case1,
    )
    np.savez(
        "./inputs_case2.npz",
        shape=shape_case2,
        fill_value=fill_value_case2,
        dtype=dtype_case2,
    )


class TestFillConstantDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs()
        shape_torch, fill_value_torch, dtype_torch = self.gen_torch_inputs()
        out_torch = self.cal_torch_res(
            shape_torch, fill_value_torch, dtype_torch
        )
        del shape_torch
        del fill_value_torch
        del dtype_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp32.npz"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_shape = np_inputs_array["shape"]
        self.np_fill_value = np_inputs_array["fill_value"]
        self.np_dtype = np_inputs_array["dtype"]

    def gen_torch_inputs(self):
        shape_torch = tuple(self.np_shape)
        fill_value_torch = torch.tensor(self.np_fill_value)
        dtype_torch = convert_dtype_to_torch_type(self.dtype)
        return shape_torch, fill_value_torch, dtype_torch

    def gen_eager_inputs(self):
        shape_eager = paddle.to_tensor(self.np_shape)
        fill_value_eager = self.np_fill_value
        dtype_eager = self.dtype
        return shape_eager, fill_value_eager, dtype_eager

    def gen_static_inputs(self):
        shape_static = paddle.to_tensor(self.np_shape)
        fill_value_static = self.np_fill_value
        dtype_static = self.dtype
        return shape_static, fill_value_static, dtype_static

    def cal_torch_res(self, shape, fill_value, dtype):
        out = torch.full(size=shape, fill_value=fill_value, dtype=dtype)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self, shape, fill_value, dtype):
        out = paddle.tensor.fill_constant(shape, dtype, fill_value)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def cal_static_res(self, shape, fill_value, dtype):
        out = paddle.tensor.fill_constant(shape, dtype, fill_value)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        shape_eager, fill_value_eager, dtype_eager = self.gen_eager_inputs()

        out_eager = self.cal_eager_res(
            shape_eager, fill_value_eager, dtype_eager
        )
        del shape_eager
        del fill_value_eager
        del dtype_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        del out_eager
        paddle.device.cuda.empty_cache()
        # save eager res for test_full_incubate
        np.savez(self.save_eager_res_path, out_eager=out_eager_np)

        # compare eager res with torch
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
            api="fill_constant",
        )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    shape_static,
                    fill_value_static,
                    dtype_static,
                ) = self.gen_static_inputs()
                out_static = self.cal_static_res(
                    shape_static,
                    fill_value_static,
                    dtype_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                fetch_list=[out_static],
            )
            out_static = out[0]

        # save static res for test_full_incubate
        np.savez(self.save_static_res_path, out_static=out_static)

        # compare static res with torch
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
            api="fill_constant",
        )

    def test_eager_stability(self):
        shape_eager, fill_value_eager, dtype_eager = self.gen_eager_inputs()
        out_eager_baseline = self.cal_eager_res(
            shape_eager, fill_value_eager, dtype_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager = self.cal_eager_res(
                shape_eager, fill_value_eager, dtype_eager
            )
            out_eager = out_eager.numpy()
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="fill_constant",
            )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    shape_static,
                    fill_value_static,
                    dtype_static,
                ) = self.gen_static_inputs()
                out_static_pg = self.cal_static_res(
                    shape_static,
                    fill_value_static,
                    dtype_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                fetch_list=[out_static_pg],
            )
            out_static_baseline = out[0]
            for i in range(50):
                out = exe.run(
                    mp,
                    fetch_list=[out_static_pg],
                )
                out_static = out[0]
                np_assert_staility(
                    out_static,
                    out_static_baseline,
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="fill_constant",
                )


class TestFillConstantDevelopCase1_FP16(TestFillConstantDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"


class TestFillConstantDevelopCase1_BF16(TestFillConstantDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"


class TestFillConstantDevelopCase2_FP32(TestFillConstantDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp32.npz"


class TestFillConstantDevelopCase2_FP16(TestFillConstantDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp16.npz"


class TestFillConstantDevelopCase2_BF16(TestFillConstantDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npz"

# (shape, dtype, value, force_cpu)
all_case = {
    # MP1
    (  (1, 70), 'int64',  -100, False,  ), 
    (  (1, 33), 'int64',  0, False,  ), 
    (  (41), 'int64',  0.0, False,  ), 
    (  (1, 26), 'int64',  -100, False,  ), 
    (  (31), 'int64',  0.0, False,  ), 
    (  (1), 'float32',  5.0, False,  ), 
    (  (1, 12), 'int64',  0, False,  ), 
    (  (1, 20), 'int64',  -100, False,  ), 
    (  (1, 6), 'int64',  0, False,  ), 
    (  (1, 1050), 'int64',  0, False,  ), 
    (  (76), 'int64',  0.0, False,  ), 
    (  (1, 64), 'int64',  -100, False,  ), 
    (  (1), 'int32',  0.0, False,  ), 
    (  (1, 356), 'int64',  0, False,  ), 
    (  (1), 'int64',  1,  True,  ), 
    (  (1, 55), 'int64',  0.0, False,  ), 
    (  (1, 1057), 'int64',  0, False,  ), 
    (  (1, 1071), 'int64',  -100, False,  ), 
    (  (1, 1099), 'int64',  0, False,  ), 
    (  (1, 1036), 'int64',  0, False,  ), 
    (  (1061), 'float32',  -100, False,  ), 
    (  (1, 1064), 'int64',  -100, False,  ), 
    (  (1, 1054), 'int64',  -100, False,  ), 
    (  (1078), 'float32',  49408, False,  ), 
    (  (1036), 'int64',  0, False,  ), 
    (  (1069), 'float32',  49408, False,  ), 
    (  (1, 22), 'int64',  -100, False,  ), 
    (  (1, 1049), 'int64',  -100, False,  ), 
    (  (1, 3), 'int64',  0, False,  ), 
    (  (1053), 'float32',  49408, False,  ), 
    (  (53), 'int64',  0.0, False,  ), 
    (  (1, 62), 'int64',  0, False,  ), 
    (  (27), 'int64',  0.0, False,  ), 
    (  (1, 20), 'int64',  0.0, False,  ), 
    (  (1039), 'float32',  49408, False,  ), 
    (  (1, 34), 'int64',  0, False,  ), 
    (  (1, 27), 'int64',  -100, False,  ), 
    (  (13), 'int64',  0.0, False,  ), 
    (  (1, 18), 'int64',  -100, False,  ), 
    (  (1038), 'float32',  -100, False,  ), 
    (  (1, 44), 'int64',  0, False,  ), 
    (  (1, 30), 'int64',  0.0, False,  ), 
    (  (1, 31), 'int64',  0.0, False,  ), 
    (  (1052), 'float32',  -100, False,  ), 
    (  (1, 1034), 'int64',  0, False,  ), 
    (  (1057), 'int64',  0, False,  ), 
    (  (1, 33), 'int64',  -100, False,  ), 
    (  (1, 41), 'int64',  0.0, False,  ), 
    (  (26), 'int64',  0.0, False,  ), 
    (  (1, 58), 'int64',  0, False,  ), 
    (  (1, 1051), 'int64',  0, False,  ), 
    (  (1, 29), 'int64',  -100, False,  ), 
    (  (1, 1038), 'int64',  -100, False,  ), 
    (  (1055), 'float32',  49408, False,  ), 
    (  (1051), 'float32',  -100, False,  ), 
    (  (1, 66), 'int64',  0, False,  ), 
    (  (1045), 'float32',  49408, False,  ), 
    (  (14), 'int64',  0.0, False,  ), 
    (  (1, 1066), 'int64',  0, False,  ), 
    (  (1, 1, 5120), 'float32',  0.0, False,  ), 
    (  (1, 1050), 'int64',  -100, False,  ), 
    (  (1), 'float32',  1,  True,  ), 
    (  (1057), 'float32',  -100, False,  ), 
    (  (1, 73), 'int64',  -100, False,  ), 
    (  (1061), 'float32',  49408, False,  ), 
    (  (1, 1043), 'int64',  -100, False,  ), 
    (  (1039), 'float32',  -100, False,  ), 
    (  (1, 53), 'int64',  0, False,  ), 
    (  (1, 14), 'int64',  -100, False,  ), 
    (  (1066), 'int64',  0, False,  ), 
    (  (1), 'int64',  128,  True,  ), 
    (  (1, 1069), 'int64',  0, False,  ), 
    (  (1, 71), 'int64',  0, False,  ), 
    (  (39), 'int64',  0.0, False,  ), 
    (  (1037), 'int64',  0, False,  ), 
    (  (1, 9), 'int64',  0, False,  ), 
    (  (1036), 'float32',  49408, False,  ), 
    (  (1, 1090), 'int64',  -100, False,  ), 
    (  (1076), 'float32',  -100, False,  ), 
    (  (36), 'int64',  0.0, False,  ), 
    (  (1, 1056), 'int64',  0, False,  ), 
    (  (1064), 'float32',  49408, False,  ), 
    (  (1050), 'float32',  49408, False,  ), 
    (  (1, 14), 'int64',  0, False,  ), 
    (  (1, 65), 'int64',  0, False,  ), 
    (  (1035), 'float32',  -100, False,  ), 
    (  (1), 'float32',  0,  True,  ), 
    (  (1, 76), 'int64',  -100, False,  ), 
    (  (1), 'int64',  64,  True,  ), 
    (  (1, 1077), 'int64',  -100, False,  ), 
    (  (1, 69), 'int64',  0, False,  ), 
    (  (1049), 'float32',  49408, False,  ), 
    (  (1, 1045), 'int64',  0, False,  ), 
    (  (1, 23), 'int64',  -100, False,  ), 
    (  (1051), 'float32',  49408, False,  ), 
    (  (1, 25), 'int64',  0.0, False,  ), 
    (  (1, 27), 'int64',  0.0, False,  ), 
    (  (1, 1090), 'int64',  0, False,  ), 
    (  (1, 1044), 'int64',  0, False,  ), 
    (  (1, 28), 'int64',  -100, False,  ), 
    (  (1, 54), 'int64',  0, False,  ), 
    (  (1, 11), 'int64',  0, False,  ), 
    (  (1, 18), 'int64',  0.0, False,  ), 
    (  (1065), 'int64',  0, False,  ), 
    (  (55), 'int64',  0.0, False,  ), 
    (  (1, 1067), 'int64',  0, False,  ), 
    (  (30), 'int64',  0.0, False,  ), 
    (  (202573824), 'float32',  0.0, False,  ), 
    (  (206063616), 'float32',  0.0, False,  ), 
    (  (1, 52), 'int64',  0, False,  ), 
    (  (1, 35), 'int64',  -100, False,  ), 
    (  (11), 'int64',  0.0, False,  ), 
    (  (1, 36), 'int64',  0, False,  ), 
    (  (1), 'int64',  0.0, False,  ), 
    (  (1, 1053), 'int64',  0, False,  ), 
    (  (1, 1041), 'int64',  -100, False,  ), 
    (  (1, 1059), 'int64',  0, False,  ), 
    (  (1, 1042), 'int64',  0, False,  ), 
    (  (1, 1053), 'int64',  -100, False,  ), 
    (  (1062), 'float32',  49408, False,  ), 
    (  (1, 46), 'int64',  0.0, False,  ), 
    (  (1, 16), 'int64',  0.0, False,  ), 
    (  (22), 'int64',  0.0, False,  ), 
    (  (1379), 'int64',  0, False,  ), 
    (  (1, 39), 'int64',  0, False,  ), 
    (  (1, 1041), 'int64',  0, False,  ), 
    (  (1, 53), 'int64',  -100, False,  ), 
    (  (1, 14), 'int64',  0.0, False,  ), 
    (  (1, 1037), 'int64',  -100, False,  ), 
    (  (1, 57), 'int64',  -100, False,  ), 
    (  (1, 1061), 'int64',  -100, False,  ), 
    (  (1), 'int64',  2,  True,  ), 
    (  (1090), 'float32',  49408, False,  ), 
    (  (1, 43), 'int64',  -100, False,  ), 
    (  (1, 42), 'int64',  -100, False,  ), 
    (  (1, 1046), 'int64',  -100, False,  ), 
    (  (1, 37), 'int64',  0, False,  ), 
    (  (1, 67), 'int64',  -100, False,  ), 
    (  (1, 1058), 'int64',  0, False,  ), 
    (  (1, 55), 'int64',  -100, False,  ), 
    (  (1061), 'int64',  0, False,  ), 
    (  (1, 5), 'int64',  0, False,  ), 
    (  (253280256), 'bfloat16',  0.0, False,  ), 
    (  (1037), 'float32',  49408, False,  ), 
    (  (1050), 'float32',  -100, False,  ), 
    (  (1, 17), 'int64',  0, False,  ), 
    (  (1, 20), 'int64',  0, False,  ), 
    (  (1, 1048), 'int64',  0, False,  ), 
    (  (1, 27), 'int64',  0, False,  ), 
    (  (1, 46), 'int64',  0, False,  ), 
    (  (1040), 'float32',  49408, False,  ), 
    (  (12), 'int64',  0.0, False,  ), 
    (  (1, 1045), 'int64',  -100, False,  ), 
    (  (1059), 'int64',  0, False,  ), 
    (  (1, 17), 'int64',  -100, False,  ), 
    (  (1090), 'float32',  -100, False,  ), 
    (  (44), 'int64',  0.0, False,  ), 
    (  (1077), 'float32',  -100, False,  ), 
    (  (1, 1078), 'int64',  -100, False,  ), 
    (  (1053), 'float32',  -100, False,  ), 
    (  (1078), 'int64',  0, False,  ), 
    (  (1, 24), 'int64',  0, False,  ), 
    (  (1048), 'int64',  0, False,  ), 
    (  (1), 'float32',  0.0,  True,  ), 
    (  (1035), 'float32',  49408, False,  ), 
    (  (1, 356), 'int64',  0.0, False,  ), 
    (  (1038), 'float32',  49408, False,  ), 
    (  (1048), 'float32',  49408, False,  ), 
    (  (1), 'float32',  32768,  True,  ), 
    (  (1, 53), 'int64',  0.0, False,  ), 
    (  (1, 72), 'int64',  0, False,  ), 
    (  (1050), 'int64',  0, False,  ), 
    (  (1, 30), 'int64',  0, False,  ), 
    (  (1, 35), 'int64',  0, False,  ), 
    (  (1034), 'float32',  49408, False,  ), 
    (  (1, 19), 'int64',  0, False,  ), 
    (  (1041), 'float32',  49408, False,  ), 
    (  (1, 48), 'int64',  -100, False,  ), 
    (  (1048), 'float32',  -100, False,  ), 
    (  (1049), 'float32',  -100, False,  ), 
    (  (1067), 'float32',  49408, False,  ), 
    (  (1), 'int32',  2,  True,  ), 
    (  (1041), 'float32',  -100, False,  ), 
    (  (1, 1), 'int64',  1024, False,  ), 
    (  (206063616), 'bfloat16',  0.0, False,  ), 
    (  (1, 1047), 'int64',  -100, False,  ), 
    (  (1, 32), 'int64',  0, False,  ), 
    (  (1, 1379), 'int64',  -100, False,  ), 
    (  (1, 71), 'int64',  -100, False,  ), 
    (  (1051), 'int64',  0, False,  ), 
    (  (1, 1048), 'int64',  -100, False,  ), 
    (  (202573824), 'bfloat16',  0.0, False,  ), 
    (  (1, 1076), 'int64',  0, False,  ), 
    (  (1065), 'float32',  -100, False,  ), 
    (  (1, 45), 'int64',  0, False,  ), 
    (  (1047), 'int64',  0, False,  ), 
    (  (1, 28), 'int64',  0.0, False,  ), 
    (  (1056), 'int64',  0, False,  ), 
    (  (1, 36), 'int64',  0.0, False,  ), 
    (  (1044), 'float32',  -100, False,  ), 
    (  (1047), 'float32',  -100, False,  ), 
    (  (1076), 'float32',  49408, False,  ), 
    (  (1, 49), 'int64',  0, False,  ), 
    (  (1, 356), 'int64',  -100, False,  ), 
    (  (1046), 'float32',  -100, False,  ), 
    (  (1038), 'int64',  0, False,  ), 
    (  (1, 16), 'int64',  -100, False,  ), 
    (  (1, 1047), 'int64',  0, False,  ), 
    (  (1071), 'float32',  49408, False,  ), 
    (  (1, 1052), 'int64',  -100, False,  ), 
    (  (1379), 'float32',  49408, False,  ), 
    (  (1066), 'float32',  -100, False,  ), 
    (  (1069), 'float32',  -100, False,  ), 
    (  (1, 21), 'int64',  0.0, False,  ), 
    (  (1, 1066), 'int64',  -100, False,  ), 
    (  (1, 48), 'int64',  0.0, False,  ), 
    (  (1055), 'float32',  -100, False,  ), 
    (  (1041), 'int64',  0, False,  ), 
    (  (1049), 'int64',  0, False,  ), 
    (  (1, 65), 'int64',  -100, False,  ), 
    (  (18), 'int64',  0.0, False,  ), 
    (  (1, 1064), 'int64',  0, False,  ), 
    (  (1, 21), 'int64',  -100, False,  ), 
    (  (1, 1049), 'int64',  0, False,  ), 
    (  (1, 63), 'int64',  -100, False,  ), 
    (  (1, 10), 'int64',  0, False,  ), 
    (  (1, 1046), 'int64',  0, False,  ), 
    (  (1, 1052), 'int64',  0, False,  ), 
    (  (1, 1025, 5120), 'float32',  0.0, False,  ), 
    (  (1062), 'int64',  0, False,  ), 
    (  (1042), 'int64',  0, False,  ), 
    (  (1, 2066), 'int64',  -100, False,  ), 
    (  (48), 'int64',  0.0, False,  ), 
    (  (1, 23), 'int64',  0.0, False,  ), 
    (  (1, 12), 'int64',  -100, False,  ), 
    (  (1, 1059), 'int64',  -100, False,  ), 
    (  (1043), 'float32',  -100, False,  ), 
    (  (1064), 'float32',  -100, False,  ), 
    (  (1, 41), 'int64',  0, False,  ), 
    (  (1, 54), 'int64',  -100, False,  ), 
    (  (1045), 'float32',  -100, False,  ), 
    (  (25), 'int64',  0.0, False,  ), 
    (  (1058), 'float32',  -100, False,  ), 
    (  (1056), 'float32',  -100, False,  ), 
    (  (1, 75), 'int64',  -100, False,  ), 
    (  (1040), 'float32',  -100, False,  ), 
    (  (1, 67), 'int64',  0.0, False,  ), 
    (  (1, 38), 'int64',  -100, False,  ), 
    (  (1039), 'int64',  0, False,  ), 
    (  (1042), 'float32',  49408, False,  ), 
    (  (1040), 'int64',  0, False,  ), 
    (  (1, 69), 'int64',  -100, False,  ), 
    (  (1, 48), 'int64',  0, False,  ), 
    (  (1054), 'float32',  49408, False,  ), 
    (  (1037), 'float32',  -100, False,  ), 
    (  (1, 1034), 'int64',  -100, False,  ), 
    (  (1036), 'float32',  -100, False,  ), 
    (  (1, 32), 'int64',  -100, False,  ), 
    (  (1035), 'int64',  0, False,  ), 
    (  (1, 1065), 'int64',  0, False,  ), 
    (  (1, 1035), 'int64',  0, False,  ), 
    (  (1078), 'float32',  -100, False,  ), 
    (  (1, 13), 'int64',  -100, False,  ), 
    (  (1059), 'float32',  -100, False,  ), 
    (  (1, 76), 'int64',  0.0, False,  ), 
    (  (1, 1039), 'int64',  0, False,  ), 
    (  (1, 1044), 'int64',  -100, False,  ), 
    (  (1), 'float32',  1.0, False,  ), 
    (  (1, 33), 'int64',  0.0, False,  ), 
    (  (46), 'int64',  0.0, False,  ), 
    (  (1, 13), 'int64',  0, False,  ), 
    (  (67), 'int64',  0.0, False,  ), 
    (  (1, 43), 'int64',  0, False,  ), 
    (  (1, 1038), 'int64',  0, False,  ), 
    (  (1077), 'float32',  49408, False,  ), 
    (  (1, 8), 'int64',  0, False,  ), 
    (  (16), 'int64',  0.0, False,  ), 
    (  (1, 59), 'int64',  0, False,  ), 
    (  (1, 39), 'int64',  -100, False,  ), 
    (  (1, 72), 'int64',  -100, False,  ), 
    (  (10487808), 'float32',  0.0, False,  ), 
    (  (1065), 'float32',  49408, False,  ), 
    (  (1, 1379), 'int64',  0, False,  ), 
    (  (209760256), 'float32',  0.0, False,  ), 
    (  (1077), 'int64',  0, False,  ), 
    (  (1, 66), 'int64',  -100, False,  ), 
    (  (138981376), 'float32',  0.0, False,  ), 
    (  (23), 'int64',  0.0, False,  ), 
    (  (1052), 'int64',  0, False,  ), 
    (  (1, 28), 'int64',  0, False,  ), 
    (  (1034), 'int64',  0, False,  ), 
    (  (1059), 'float32',  49408, False,  ), 
    (  (21), 'int64',  0.0, False,  ), 
    (  (1, 34), 'int64',  0.0, False,  ), 
    (  (38), 'int64',  0.0, False,  ), 
    (  (1, 26), 'int64',  0, False,  ), 
    (  (1, 51), 'int64',  0, False,  ), 
    (  (1, 24), 'int64',  0.0, False,  ), 
    (  (1, 1065), 'int64',  -100, False,  ), 
    (  (157321216), 'float32',  0.0, False,  ), 
    (  (1, 15), 'int64',  -100, False,  ), 
    (  (1, 11), 'int64',  0.0, False,  ), 
    (  (29), 'int64',  0.0, False,  ), 
    (  (157321216), 'bfloat16',  0.0, False,  ), 
    (  (1, 1057), 'int64',  -100, False,  ), 
    (  (1, 1055), 'int64',  0, False,  ), 
    (  (1076), 'int64',  0, False,  ), 
    (  (1, 25), 'int64',  -100, False,  ), 
    (  (1052), 'float32',  49408, False,  ), 
    (  (1058), 'int64',  0, False,  ), 
    (  (1, 16), 'int64',  0, False,  ), 
    (  (1, 1055), 'int64',  -100, False,  ), 
    (  (1071), 'float32',  -100, False,  ), 
    (  (17), 'int64',  0.0, False,  ), 
    (  (1, 1051), 'int64',  -100, False,  ), 
    (  (10487808), 'bfloat16',  0.0, False,  ), 
    (  (1, 68), 'int64',  0, False,  ), 
    (  (1, 64), 'int64',  0, False,  ), 
    (  (1, 62), 'int64',  -100, False,  ), 
    (  (1, 1043), 'int64',  0, False,  ), 
    (  (1, 57), 'int64',  0, False,  ), 
    (  (24), 'int64',  0.0, False,  ), 
    (  (1, 52), 'int64',  -100, False,  ), 
    (  (1, 40), 'int64',  0, False,  ), 
    (  (1, 22), 'int64',  0.0, False,  ), 
    (  (20), 'int64',  0.0, False,  ), 
    (  (54), 'int64',  0.0, False,  ), 
    (  (2), 'int64',  0.0, False,  ), 
    (  (1, 1040), 'int64',  0, False,  ), 
    (  (1071), 'int64',  0, False,  ), 
    (  (1, 39), 'int64',  0.0, False,  ), 
    (  (1, 22), 'int64',  0, False,  ), 
    (  (1, 29), 'int64',  0, False,  ), 
    (  (1, 43), 'int64',  0.0, False,  ), 
    (  (1042), 'float32',  -100, False,  ), 
    (  (1, 38), 'int64',  0, False,  ), 
    (  (1, 31), 'int64',  -100, False,  ), 
    (  (1, 67), 'int64',  0, False,  ), 
    (  (1, 1058), 'int64',  -100, False,  ), 
    (  (34), 'int64',  0.0, False,  ), 
    (  (35), 'int64',  0.0, False,  ), 
    (  (1, 58), 'int64',  -100, False,  ), 
    (  (1, 1040), 'int64',  -100, False,  ), 
    (  (1, 74), 'int64',  0, False,  ), 
    (  (42), 'int64',  0.0, False,  ), 
    (  (32), 'int64',  0.0, False,  ), 
    (  (1, 51), 'int64',  -100, False,  ), 
    (  (1, 76), 'int64',  0, False,  ), 
    (  (1, 26), 'int64',  0.0, False,  ), 
    (  (138981376), 'bfloat16',  0.0, False,  ), 
    (  (1067), 'int64',  0, False,  ), 
    (  (1, 13), 'int64',  0.0, False,  ), 
    (  (1, 1037), 'int64',  0, False,  ), 
    (  (1, 1062), 'int64',  0, False,  ), 
    (  (1, 24), 'int64',  -100, False,  ), 
    (  (19), 'int64',  0.0, False,  ), 
    (  (1, 47), 'int64',  -100, False,  ), 
    (  (1, 32), 'int64',  0.0, False,  ), 
    (  (1, 18), 'int64',  0, False,  ), 
    (  (1, 55), 'int64',  0, False,  ), 
    (  (1, 74), 'int64',  -100, False,  ), 
    (  (253280256), 'float32',  0.0, False,  ), 
    (  (1, 73), 'int64',  0, False,  ), 
    (  (1056), 'float32',  49408, False,  ), 
    (  (1, 1076), 'int64',  -100, False,  ), 
    (  (1, 11), 'int64',  -100, False,  ), 
    (  (1069), 'int64',  0, False,  ), 
    (  (1, 63), 'int64',  0, False,  ), 
    (  (1, 1035), 'int64',  -100, False,  ), 
    (  (1, 44), 'int64',  -100, False,  ), 
    (  (1, 1062), 'int64',  -100, False,  ), 
    (  (1099), 'float32',  -100, False,  ), 
    (  (1, 17), 'int64',  0.0, False,  ), 
    (  (1043), 'int64',  0, False,  ), 
    (  (1067), 'float32',  -100, False,  ), 
    (  (1, 34), 'int64',  -100, False,  ), 
    (  (1, 68), 'int64',  -100, False,  ), 
    (  (1, 35), 'int64',  0.0, False,  ), 
    (  (1379), 'float32',  -100, False,  ), 
    (  (1054), 'float32',  -100, False,  ), 
    (  (1, 1061), 'int64',  0, False,  ), 
    (  (209760256), 'bfloat16',  0.0, False,  ), 
    (  (1, 30), 'int64',  -100, False,  ), 
    (  (1, 1077), 'int64',  0, False,  ), 
    (  (1, 1071), 'int64',  0, False,  ), 
    (  (1, 19), 'int64',  0.0, False,  ), 
    (  (1055), 'int64',  0, False,  ), 
    (  (1, 1042), 'int64',  -100, False,  ), 
    (  (1, 1099), 'int64',  -100, False,  ), 
    (  (1046), 'float32',  49408, False,  ), 
    (  (1066), 'float32',  49408, False,  ), 
    (  (28), 'int64',  0.0, False,  ), 
    (  (1, 1039), 'int64',  -100, False,  ), 
    (  (1047), 'float32',  49408, False,  ), 
    (  (15), 'int64',  0.0, False,  ), 
    (  (1, 7), 'int64',  0, False,  ), 
    (  (1099), 'float32',  49408, False,  ), 
    (  (1, 31), 'int64',  0, False,  ), 
    (  (1, 12), 'int64',  0.0, False,  ), 
    (  (1, 1067), 'int64',  -100, False,  ), 
    (  (1057), 'float32',  49408, False,  ), 
    (  (1045), 'int64',  0, False,  ), 
    (  (1062), 'float32',  -100, False,  ), 
    (  (1, 42), 'int64',  0.0, False,  ), 
    (  (1034), 'float32',  -100, False,  ), 
    (  (1, 1069), 'int64',  -100, False,  ), 
    (  (1, 46), 'int64',  -100, False,  ), 
    (  (1043), 'float32',  49408, False,  ), 
    (  (1, 41), 'int64',  -100, False,  ), 
    (  (1054), 'int64',  0, False,  ), 
    (  (1044), 'int64',  0, False,  ), 
    (  (1044), 'float32',  49408, False,  ), 
    (  (1099), 'int64',  0, False,  ), 
    (  (1090), 'int64',  0, False,  ), 
    (  (1, 1078), 'int64',  0, False,  ), 
    (  (1), 'int64',  0,  True,  ), 
    (  (1, 1036), 'int64',  -100, False,  ), 
    (  (1, 49), 'int64',  -100, False,  ), 
    (  (1, 47), 'int64',  0, False,  ), 
    (  (1, 1056), 'int64',  -100, False,  ), 
    (  (1058), 'float32',  49408, False,  ), 
    (  (1, 29), 'int64',  0.0, False,  ), 
    (  (1, 23), 'int64',  0, False,  ), 
    (  (1, 21), 'int64',  0, False,  ), 
    (  (1, 19), 'int64',  -100, False,  ), 
    (  (1, 15), 'int64',  0.0, False,  ), 
    (  (33), 'int64',  0.0, False,  ), 
    (  (1, 15), 'int64',  0, False,  ), 
    (  (43), 'int64',  0.0, False,  ), 
    (  (1, 36), 'int64',  -100, False,  ), 
    (  (1053), 'int64',  0, False,  ), 
    (  (1, 1054), 'int64',  0, False,  ), 
    (  (1, 44), 'int64',  0.0, False,  ), 
    (  (1046), 'int64',  0, False,  ), 
    (  (356), 'int64',  0.0, False,  ), 
    (  (1, 75), 'int64',  0, False,  ), 
    (  (1064), 'int64',  0, False,  ), 
    (  (1, 54), 'int64',  0.0, False,  ), 
    (  (1, 42), 'int64',  0, False,  ), 
    (  (1, 2066), 'int64',  0, False,  ), 
    (  (1, 4), 'int64',  0, False,  ), 
    (  (1, 38), 'int64',  0.0, False,  ), 
    (  (1, 70), 'int64',  0, False,  ), 
    (  (1, 25), 'int64',  0, False,  ), 
    # MP8
    (  (33), 'int64',  0.0, False,  ), 
    (  (1), 'float32',  0,  True,  ), 
    (  (1, 1354), 'int64',  -100, False,  ), 
    (  (1, 1063), 'int64',  -100, False,  ), 
    (  (24), 'int64',  0.0, False,  ), 
    (  (1, 68), 'int64',  0, False,  ), 
    (  (1, 2339), 'int64',  -100, False,  ), 
    (  (1035), 'int64',  0, False,  ), 
    (  (1), 'int64',  1024,  True,  ), 
    (  (202573824), 'bfloat16',  0.0, False,  ), 
    (  (2983), 'float32',  -100, False,  ), 
    (  (1040), 'int64',  0, False,  ), 
    (  (1, 58), 'int64',  0, False,  ), 
    (  (1, 50), 'int64',  -100, False,  ), 
    (  (1, 29), 'int64',  -100, False,  ), 
    (  (1, 14), 'int64',  -100, False,  ), 
    (  (1), 'int32',  2,  True,  ), 
    (  (1078), 'float32',  -100, False,  ), 
    (  (1, 12), 'int64',  0.0, False,  ), 
    (  (1, 1055), 'int64',  -100, False,  ), 
    (  (1, 2745), 'int64',  0, False,  ), 
    (  (2377), 'float32',  49408, False,  ), 
    (  (1, 1070), 'int64',  0, False,  ), 
    (  (1, 1264), 'int64',  0, False,  ), 
    (  (1070), 'int64',  0, False,  ), 
    (  (1, 15), 'int64',  -100, False,  ), 
    (  (1, 21), 'int64',  -100, False,  ), 
    (  (1722), 'int64',  0.0, False,  ), 
    (  (1, 3041), 'int64',  0, False,  ), 
    (  (2745), 'int64',  0, False,  ), 
    (  (1, 1067), 'int64',  0, False,  ), 
    (  (1, 36), 'int64',  -100, False,  ), 
    (  (1, 43), 'int64',  0, False,  ), 
    (  (1, 1072), 'int64',  0, False,  ), 
    (  (140733440), 'float32',  0.0, False,  ), 
    (  (1, 10), 'int64',  0, False,  ), 
    (  (1061), 'float32',  49408, False,  ), 
    (  (1051), 'float32',  49408, False,  ), 
    (  (1), 'int64',  1,  True,  ), 
    (  (1058), 'float32',  -100, False,  ), 
    (  (1, 16), 'int64',  0, False,  ), 
    (  (1055), 'float32',  -100, False,  ), 
    (  (1, 1890), 'int64',  0, False,  ), 
    (  (1, 29), 'int64',  0.0, False,  ), 
    (  (1, 1036), 'int64',  -100, False,  ), 
    (  (1, 52), 'int64',  -100, False,  ), 
    (  (1, 2913), 'int64',  0, False,  ), 
    (  (1, 1057), 'int64',  0, False,  ), 
    (  (1, 41), 'int64',  -100, False,  ), 
    (  (54), 'int64',  0.0, False,  ), 
    (  (1, 60), 'int64',  -100, False,  ), 
    (  (26), 'int64',  0.0, False,  ), 
    (  (1, 45), 'int64',  0, False,  ), 
    (  (1, 1041), 'int64',  -100, False,  ), 
    (  (1043), 'float32',  49408, False,  ), 
    (  (1050), 'int64',  0, False,  ), 
    (  (30), 'int64',  0.0, False,  ), 
    (  (1, 20), 'int64',  0.0, False,  ), 
    (  (1044), 'int64',  0, False,  ), 
    (  (1, 48), 'int64',  0.0, False,  ), 
    (  (1, 1), 'int64',  1024, False,  ), 
    (  (1, 18), 'int64',  -100, False,  ), 
    (  (1, 2377), 'int64',  0, False,  ), 
    (  (1, 35), 'int64',  0.0, False,  ), 
    (  (1053), 'float32',  49408, False,  ), 
    (  (1, 21), 'int64',  0.0, False,  ), 
    (  (1, 27), 'int64',  0, False,  ), 
    (  (1), 'float32',  32768,  True,  ), 
    (  (1, 1034), 'int64',  -100, False,  ), 
    (  (1, 1050), 'int64',  -100, False,  ), 
    (  (1, 1060), 'int64',  0, False,  ), 
    (  (1, 38), 'int64',  -100, False,  ), 
    (  (1049), 'int64',  0, False,  ), 
    (  (1, 1354), 'int64',  0, False,  ), 
    (  (1, 26), 'int64',  0.0, False,  ), 
    (  (1, 33), 'int64',  0, False,  ), 
    (  (1, 54), 'int64',  -100, False,  ), 
    (  (1053), 'int64',  0, False,  ), 
    (  (19), 'int64',  0.0, False,  ), 
    (  (1, 8), 'int64',  0, False,  ), 
    (  (1, 2287), 'int64',  0, False,  ), 
    (  (1, 17), 'int64',  0.0, False,  ), 
    (  (1, 46), 'int64',  0, False,  ), 
    (  (44), 'int64',  0.0, False,  ), 
    (  (1038), 'float32',  -100, False,  ), 
    (  (1, 32), 'int64',  0, False,  ), 
    (  (1047), 'int64',  0, False,  ), 
    (  (1, 1053), 'int64',  -100, False,  ), 
    (  (1, 31), 'int64',  0.0, False,  ), 
    (  (1, 37), 'int64',  -100, False,  ), 
    (  (1, 1890), 'int64',  0.0, False,  ), 
    (  (2018), 'int64',  0.0, False,  ), 
    (  (1, 1066), 'int64',  -100, False,  ), 
    (  (1, 1035), 'int64',  -100, False,  ), 
    (  (1, 30), 'int64',  0, False,  ), 
    (  (1, 53), 'int64',  -100, False,  ), 
    (  (1081), 'float32',  49408, False,  ), 
    (  (1, 32), 'int64',  0.0, False,  ), 
    (  (1), 'float32',  5.0, False,  ), 
    (  (1, 23), 'int64',  0, False,  ), 
    (  (1, 38), 'int64',  0, False,  ), 
    (  (1, 1058), 'int64',  -100, False,  ), 
    (  (1, 45), 'int64',  -100, False,  ), 
    (  (1048), 'float32',  49408, False,  ), 
    (  (1067), 'int64',  0, False,  ), 
    (  (1077), 'float32',  49408, False,  ), 
    (  (2913), 'float32',  -100, False,  ), 
    (  (1, 1078), 'int64',  0, False,  ), 
    (  (1, 20), 'int64',  -100, False,  ), 
    (  (1, 30), 'int64',  0.0, False,  ), 
    (  (1), 'int32',  0.0, False,  ), 
    (  (1, 22), 'int64',  0, False,  ), 
    (  (1, 4), 'int64',  0, False,  ), 
    (  (1, 52), 'int64',  0, False,  ), 
    (  (1041), 'float32',  49408, False,  ), 
    (  (1, 16), 'int64',  0.0, False,  ), 
    (  (1, 1960), 'int64',  0.0, False,  ), 
    (  (1, 1960), 'int64',  0, False,  ), 
    (  (1, 1062), 'int64',  0, False,  ), 
    (  (1066), 'float32',  -100, False,  ), 
    (  (1, 3), 'int64',  0, False,  ), 
    (  (1, 3041), 'int64',  -100, False,  ), 
    (  (1, 1066), 'int64',  0, False,  ), 
    (  (2377), 'float32',  -100, False,  ), 
    (  (1045), 'int64',  0, False,  ), 
    (  (1, 2018), 'int64',  -100, False,  ), 
    (  (1, 1062), 'int64',  -100, False,  ), 
    (  (1, 1737), 'int64',  0, False,  ), 
    (  (1, 15), 'int64',  0, False,  ), 
    (  (2287), 'float32',  -100, False,  ), 
    (  (1, 1035), 'int64',  0, False,  ), 
    (  (1, 1737), 'int64',  -100, False,  ), 
    (  (1, 1345), 'int64',  0, False,  ), 
    (  (1041), 'int64',  0, False,  ), 
    (  (1056), 'int64',  0, False,  ), 
    (  (1046), 'float32',  -100, False,  ), 
    (  (1, 38), 'int64',  0.0, False,  ), 
    (  (1, 2018), 'int64',  0.0, False,  ), 
    (  (1, 1039), 'int64',  0, False,  ), 
    (  (1, 28), 'int64',  0, False,  ), 
    (  (1034), 'float32',  49408, False,  ), 
    (  (1036), 'float32',  49408, False,  ), 
    (  (1, 66), 'int64',  -100, False,  ), 
    (  (1, 31), 'int64',  -100, False,  ), 
    (  (1, 1040), 'int64',  0, False,  ), 
    (  (1051), 'float32',  -100, False,  ), 
    (  (1), 'float32',  1,  True,  ), 
    (  (1, 1054), 'int64',  0, False,  ), 
    (  (1, 55), 'int64',  0, False,  ), 
    (  (1, 47), 'int64',  -100, False,  ), 
    (  (2913), 'float32',  49408, False,  ), 
    (  (1, 59), 'int64',  -100, False,  ), 
    (  (1), 'int64',  2,  True,  ), 
    (  (1, 1041), 'int64',  0, False,  ), 
    (  (1050), 'float32',  49408, False,  ), 
    (  (1, 44), 'int64',  0.0, False,  ), 
    (  (1043), 'float32',  -100, False,  ), 
    (  (1, 1298), 'int64',  0, False,  ), 
    (  (1, 1042), 'int64',  0, False,  ), 
    (  (1), 'int64',  64,  True,  ), 
    (  (1, 2018), 'int64',  0, False,  ), 
    (  (1060), 'float32',  49408, False,  ), 
    (  (1072), 'float32',  49408, False,  ), 
    (  (29), 'int64',  0.0, False,  ), 
    (  (1063), 'float32',  -100, False,  ), 
    (  (1, 19), 'int64',  -100, False,  ), 
    (  (1, 36), 'int64',  0, False,  ), 
    (  (39), 'int64',  0.0, False,  ), 
    (  (1, 48), 'int64',  0, False,  ), 
    (  (175423744), 'bfloat16',  0.0, False,  ), 
    (  (40), 'int64',  0.0, False,  ), 
    (  (1, 22), 'int64',  0.0, False,  ), 
    (  (1, 1043), 'int64',  -100, False,  ), 
    (  (1, 2913), 'int64',  -100, False,  ), 
    (  (1354), 'int64',  0.0, False,  ), 
    (  (1, 25), 'int64',  0.0, False,  ), 
    (  (1, 18), 'int64',  0.0, False,  ), 
    (  (1, 11), 'int64',  -100, False,  ), 
    (  (17), 'int64',  0.0, False,  ), 
    (  (1037), 'float32',  49408, False,  ), 
    (  (1057), 'float32',  49408, False,  ), 
    (  (1, 35), 'int64',  -100, False,  ), 
    (  (1, 1047), 'int64',  0, False,  ), 
    (  (1, 1290), 'int64',  -100, False,  ), 
    (  (1060), 'int64',  0, False,  ), 
    (  (1046), 'float32',  49408, False,  ), 
    (  (1), 'float32',  1.0, False,  ), 
    (  (1042), 'float32',  -100, False,  ), 
    (  (1, 23), 'int64',  0.0, False,  ), 
    (  (1040), 'float32',  -100, False,  ), 
    (  (1048), 'int64',  0, False,  ), 
    (  (1, 1049), 'int64',  0, False,  ), 
    (  (1, 1345), 'int64',  -100, False,  ), 
    (  (1), 'float32',  0.0,  True,  ), 
    (  (1039), 'int64',  0, False,  ), 
    (  (1046), 'int64',  0, False,  ), 
    (  (1, 1036), 'int64',  0, False,  ), 
    (  (34), 'int64',  0.0, False,  ), 
    (  (2339), 'float32',  -100, False,  ), 
    (  (1, 40), 'int64',  0, False,  ), 
    (  (55), 'int64',  0.0, False,  ), 
    (  (1, 1932), 'int64',  -100, False,  ), 
    (  (1, 2377), 'int64',  -100, False,  ), 
    (  (15), 'int64',  0.0, False,  ), 
    (  (2955), 'float32',  -100, False,  ), 
    (  (1, 56), 'int64',  0, False,  ), 
    (  (1, 40), 'int64',  0.0, False,  ), 
    (  (2745), 'float32',  -100, False,  ), 
    (  (1072), 'int64',  0, False,  ), 
    (  (1, 2955), 'int64',  0, False,  ), 
    (  (1048), 'float32',  -100, False,  ), 
    (  (1, 1298), 'int64',  -100, False,  ), 
    (  (1049), 'float32',  49408, False,  ), 
    (  (1, 62), 'int64',  -100, False,  ), 
    (  (1, 28), 'int64',  0.0, False,  ), 
    (  (1044), 'float32',  49408, False,  ), 
    (  (1077), 'int64',  0, False,  ), 
    (  (1054), 'float32',  -100, False,  ), 
    (  (1, 56), 'int64',  -100, False,  ), 
    (  (1044), 'float32',  -100, False,  ), 
    (  (1, 48), 'int64',  -100, False,  ), 
    (  (1, 27), 'int64',  0.0, False,  ), 
    (  (1, 42), 'int64',  -100, False,  ), 
    (  (1067), 'float32',  -100, False,  ), 
    (  (1, 61), 'int64',  -100, False,  ), 
    (  (1, 2287), 'int64',  -100, False,  ), 
    (  (1316), 'int64',  0.0, False,  ), 
    (  (1, 1043), 'int64',  0, False,  ), 
    (  (1, 61), 'int64',  0, False,  ), 
    (  (1, 60), 'int64',  0, False,  ), 
    (  (1, 2745), 'int64',  -100, False,  ), 
    (  (1, 1070), 'int64',  -100, False,  ), 
    (  (1, 1, 5120), 'float32',  0.0, False,  ), 
    (  (1071), 'float32',  -100, False,  ), 
    (  (1, 44), 'int64',  0, False,  ), 
    (  (202573824), 'float32',  0.0, False,  ), 
    (  (1, 1316), 'int64',  -100, False,  ), 
    (  (1062), 'int64',  0, False,  ), 
    (  (1, 1052), 'int64',  0, False,  ), 
    (  (1059), 'int64',  0, False,  ), 
    (  (1052), 'int64',  0, False,  ), 
    (  (1078), 'float32',  49408, False,  ), 
    (  (13), 'int64',  0.0, False,  ), 
    (  (1, 22), 'int64',  -100, False,  ), 
    (  (36), 'int64',  0.0, False,  ), 
    (  (16), 'int64',  0.0, False,  ), 
    (  (1056), 'float32',  49408, False,  ), 
    (  (1, 31), 'int64',  0, False,  ), 
    (  (1, 1038), 'int64',  -100, False,  ), 
    (  (1, 19), 'int64',  0.0, False,  ), 
    (  (1, 7), 'int64',  0, False,  ), 
    (  (1, 76), 'int64',  -100, False,  ), 
    (  (1, 14), 'int64',  0.0, False,  ), 
    (  (1063), 'float32',  49408, False,  ), 
    (  (1, 50), 'int64',  0, False,  ), 
    (  (1, 47), 'int64',  0, False,  ), 
    (  (1, 67), 'int64',  0, False,  ), 
    (  (1, 1077), 'int64',  -100, False,  ), 
    (  (1, 73), 'int64',  0, False,  ), 
    (  (1, 1051), 'int64',  -100, False,  ), 
    (  (1, 1264), 'int64',  0.0, False,  ), 
    (  (1053), 'float32',  -100, False,  ), 
    (  (1, 1077), 'int64',  0, False,  ), 
    (  (1, 1060), 'int64',  -100, False,  ), 
    (  (1, 37), 'int64',  0, False,  ), 
    (  (1, 36), 'int64',  0.0, False,  ), 
    (  (22), 'int64',  0.0, False,  ), 
    (  (1, 1052), 'int64',  -100, False,  ), 
    (  (1071), 'int64',  0, False,  ), 
    (  (1057), 'int64',  0, False,  ), 
    (  (48), 'int64',  0.0, False,  ), 
    (  (1054), 'int64',  0, False,  ), 
    (  (1042), 'int64',  0, False,  ), 
    (  (23), 'int64',  0.0, False,  ), 
    (  (1, 40), 'int64',  -100, False,  ), 
    (  (1, 2339), 'int64',  0, False,  ), 
    (  (1058), 'float32',  49408, False,  ), 
    (  (14), 'int64',  0.0, False,  ), 
    (  (1, 44), 'int64',  -100, False,  ), 
    (  (1, 26), 'int64',  0, False,  ), 
    (  (1, 1420), 'int64',  0, False,  ), 
    (  (12), 'int64',  0.0, False,  ), 
    (  (1, 1354), 'int64',  0.0, False,  ), 
    (  (1960), 'int64',  0.0, False,  ), 
    (  (1, 1051), 'int64',  0, False,  ), 
    (  (35), 'int64',  0.0, False,  ), 
    (  (1, 2983), 'int64',  0, False,  ), 
    (  (1, 1071), 'int64',  0, False,  ), 
    (  (28), 'int64',  0.0, False,  ), 
    (  (10487808), 'bfloat16',  0.0, False,  ), 
    (  (1, 76), 'int64',  0, False,  ), 
    (  (1932), 'int64',  0.0, False,  ), 
    (  (1, 39), 'int64',  0, False,  ), 
    (  (1, 12), 'int64',  -100, False,  ), 
    (  (1, 43), 'int64',  -100, False,  ), 
    (  (1, 28), 'int64',  -100, False,  ), 
    (  (2983), 'float32',  49408, False,  ), 
    (  (1, 1881), 'int64',  -100, False,  ), 
    (  (1070), 'float32',  -100, False,  ), 
    (  (1, 55), 'int64',  -100, False,  ), 
    (  (2339), 'int64',  0, False,  ), 
    (  (1, 1722), 'int64',  0, False,  ), 
    (  (10487808), 'float32',  0.0, False,  ), 
    (  (1, 24), 'int64',  0.0, False,  ), 
    (  (1, 57), 'int64',  0, False,  ), 
    (  (1, 1055), 'int64',  0, False,  ), 
    (  (1, 1316), 'int64',  0, False,  ), 
    (  (1055), 'int64',  0, False,  ), 
    (  (140733440), 'bfloat16',  0.0, False,  ), 
    (  (1, 51), 'int64',  -100, False,  ), 
    (  (1, 1042), 'int64',  -100, False,  ), 
    (  (1, 1960), 'int64',  -100, False,  ), 
    (  (3041), 'float32',  49408, False,  ), 
    (  (1059), 'float32',  -100, False,  ), 
    (  (2287), 'int64',  0, False,  ), 
    (  (1062), 'float32',  -100, False,  ), 
    (  (1, 26), 'int64',  -100, False,  ), 
    (  (2983), 'int64',  0, False,  ), 
    (  (1, 18), 'int64',  0, False,  ), 
    (  (1, 1037), 'int64',  0, False,  ), 
    (  (1, 23), 'int64',  -100, False,  ), 
    (  (1049), 'float32',  -100, False,  ), 
    (  (1052), 'float32',  -100, False,  ), 
    (  (1042), 'float32',  49408, False,  ), 
    (  (1058), 'int64',  0, False,  ), 
    (  (1, 13), 'int64',  -100, False,  ), 
    (  (1, 1058), 'int64',  0, False,  ), 
    (  (1, 30), 'int64',  -100, False,  ), 
    (  (1, 43), 'int64',  0.0, False,  ), 
    (  (1, 1057), 'int64',  -100, False,  ), 
    (  (1, 1038), 'int64',  0, False,  ), 
    (  (1, 51), 'int64',  0, False,  ), 
    (  (1052), 'float32',  49408, False,  ), 
    (  (1, 1932), 'int64',  0.0, False,  ), 
    (  (1, 1048), 'int64',  0, False,  ), 
    (  (1059), 'float32',  49408, False,  ), 
    (  (43), 'int64',  0.0, False,  ), 
    (  (1, 1050), 'int64',  0, False,  ), 
    (  (1, 1054), 'int64',  -100, False,  ), 
    (  (1, 1264), 'int64',  -100, False,  ), 
    (  (1, 46), 'int64',  -100, False,  ), 
    (  (1, 1561), 'int64',  0, False,  ), 
    (  (1, 1815), 'int64',  0, False,  ), 
    (  (2745), 'float32',  49408, False,  ), 
    (  (1, 1059), 'int64',  0, False,  ), 
    (  (1056), 'float32',  -100, False,  ), 
    (  (1, 1722), 'int64',  0.0, False,  ), 
    (  (1, 1045), 'int64',  0, False,  ), 
    (  (1066), 'float32',  49408, False,  ), 
    (  (1077), 'float32',  -100, False,  ), 
    (  (1, 1063), 'int64',  0, False,  ), 
    (  (1, 13), 'int64',  0, False,  ), 
    (  (1034), 'float32',  -100, False,  ), 
    (  (1, 1056), 'int64',  0, False,  ), 
    (  (1, 37), 'int64',  0.0, False,  ), 
    (  (1, 1049), 'int64',  -100, False,  ), 
    (  (1, 27), 'int64',  -100, False,  ), 
    (  (1, 19), 'int64',  0, False,  ), 
    (  (1, 2955), 'int64',  -100, False,  ), 
    (  (1, 17), 'int64',  0, False,  ), 
    (  (1054), 'float32',  49408, False,  ), 
    (  (1, 1046), 'int64',  0, False,  ), 
    (  (1, 1044), 'int64',  -100, False,  ), 
    (  (1, 29), 'int64',  0, False,  ), 
    (  (1, 59), 'int64',  0, False,  ), 
    (  (1061), 'int64',  0, False,  ), 
    (  (31), 'int64',  0.0, False,  ), 
    (  (1038), 'int64',  0, False,  ), 
    (  (1071), 'float32',  49408, False,  ), 
    (  (1, 20), 'int64',  0, False,  ), 
    (  (1035), 'float32',  49408, False,  ), 
    (  (1, 1056), 'int64',  -100, False,  ), 
    (  (1, 1932), 'int64',  0, False,  ), 
    (  (1, 1072), 'int64',  -100, False,  ), 
    (  (1, 24), 'int64',  -100, False,  ), 
    (  (1036), 'float32',  -100, False,  ), 
    (  (1, 49), 'int64',  0.0, False,  ), 
    (  (1, 47), 'int64',  0.0, False,  ), 
    (  (18), 'int64',  0.0, False,  ), 
    (  (1, 1044), 'int64',  0, False,  ), 
    (  (47), 'int64',  0.0, False,  ), 
    (  (1072), 'float32',  -100, False,  ), 
    (  (1, 5), 'int64',  0, False,  ), 
    (  (1, 1039), 'int64',  -100, False,  ), 
    (  (1051), 'int64',  0, False,  ), 
    (  (1, 12), 'int64',  0, False,  ), 
    (  (1, 1053), 'int64',  0, False,  ), 
    (  (1, 1034), 'int64',  0, False,  ), 
    (  (1, 34), 'int64',  0.0, False,  ), 
    (  (1055), 'float32',  49408, False,  ), 
    (  (1039), 'float32',  49408, False,  ), 
    (  (1, 57), 'int64',  -100, False,  ), 
    (  (2913), 'int64',  0, False,  ), 
    (  (1, 11), 'int64',  0, False,  ), 
    (  (1, 34), 'int64',  -100, False,  ), 
    (  (1061), 'float32',  -100, False,  ), 
    (  (37), 'int64',  0.0, False,  ), 
    (  (32), 'int64',  0.0, False,  ), 
    (  (1057), 'float32',  -100, False,  ), 
    (  (1, 1815), 'int64',  -100, False,  ), 
    (  (1, 58), 'int64',  0.0, False,  ), 
    (  (2955), 'int64',  0, False,  ), 
    (  (1, 9), 'int64',  0, False,  ), 
    (  (2287), 'float32',  49408, False,  ), 
    (  (1, 21), 'int64',  0, False,  ), 
    (  (1, 49), 'int64',  0, False,  ), 
    (  (2377), 'int64',  0, False,  ), 
    (  (1, 1078), 'int64',  -100, False,  ), 
    (  (58), 'int64',  0.0, False,  ), 
    (  (1), 'int64',  0.0, False,  ), 
    (  (1, 53), 'int64',  0, False,  ), 
    (  (1, 54), 'int64',  0.0, False,  ), 
    (  (1034), 'int64',  0, False,  ), 
    (  (1, 39), 'int64',  0.0, False,  ), 
    (  (1047), 'float32',  49408, False,  ), 
    (  (1, 73), 'int64',  -100, False,  ), 
    (  (175423744), 'float32',  0.0, False,  ), 
    (  (20), 'int64',  0.0, False,  ), 
    (  (27), 'int64',  0.0, False,  ), 
    (  (49), 'int64',  0.0, False,  ), 
    (  (2955), 'float32',  49408, False,  ), 
    (  (1, 33), 'int64',  0.0, False,  ), 
    (  (1, 49), 'int64',  -100, False,  ), 
    (  (1, 66), 'int64',  0, False,  ), 
    (  (1039), 'float32',  -100, False,  ), 
    (  (1, 1420), 'int64',  -100, False,  ), 
    (  (1, 1040), 'int64',  -100, False,  ), 
    (  (1, 1025, 5120), 'float32',  0.0, False,  ), 
    (  (1, 33), 'int64',  -100, False,  ), 
    (  (1, 35), 'int64',  0, False,  ), 
    (  (1, 34), 'int64',  0, False,  ), 
    (  (1038), 'float32',  49408, False,  ), 
    (  (1067), 'float32',  49408, False,  ), 
    (  (1047), 'float32',  -100, False,  ), 
    (  (1, 1081), 'int64',  -100, False,  ), 
    (  (1, 14), 'int64',  0, False,  ), 
    (  (1081), 'int64',  0, False,  ), 
    (  (3041), 'float32',  -100, False,  ), 
    (  (1045), 'float32',  -100, False,  ), 
    (  (1037), 'float32',  -100, False,  ), 
    (  (1, 6), 'int64',  0, False,  ), 
    (  (25), 'int64',  0.0, False,  ), 
    (  (2339), 'float32',  49408, False,  ), 
    (  (2), 'int64',  0.0, False,  ), 
    (  (1, 1081), 'int64',  0, False,  ), 
    (  (1050), 'float32',  -100, False,  ), 
    (  (1040), 'float32',  49408, False,  ), 
    (  (1, 1059), 'int64',  -100, False,  ), 
    (  (1062), 'float32',  49408, False,  ), 
    (  (1, 42), 'int64',  0, False,  ), 
    (  (1, 11), 'int64',  0.0, False,  ), 
    (  (1043), 'int64',  0, False,  ), 
    (  (1, 1071), 'int64',  -100, False,  ), 
    (  (1), 'int64',  0,  True,  ), 
    (  (1, 1048), 'int64',  -100, False,  ), 
    (  (1, 1561), 'int64',  -100, False,  ), 
    (  (21), 'int64',  0.0, False,  ), 
    (  (1, 1045), 'int64',  -100, False,  ), 
    (  (1, 1047), 'int64',  -100, False,  ), 
    (  (1037), 'int64',  0, False,  ), 
    (  (1264), 'int64',  0.0, False,  ), 
    (  (1, 1316), 'int64',  0.0, False,  ), 
    (  (1, 1037), 'int64',  -100, False,  ), 
    (  (1, 54), 'int64',  0, False,  ), 
    (  (1, 58), 'int64',  -100, False,  ), 
    (  (1, 32), 'int64',  -100, False,  ), 
    (  (1036), 'int64',  0, False,  ), 
    (  (3041), 'int64',  0, False,  ), 
    (  (1078), 'int64',  0, False,  ), 
    (  (1, 24), 'int64',  0, False,  ), 
    (  (1, 1061), 'int64',  -100, False,  ), 
    (  (1, 1722), 'int64',  -100, False,  ), 
    (  (1, 65), 'int64',  0, False,  ), 
    (  (1066), 'int64',  0, False,  ), 
    (  (1, 39), 'int64',  -100, False,  ), 
    (  (1081), 'float32',  -100, False,  ), 
    (  (1, 25), 'int64',  0, False,  ), 
    (  (1060), 'float32',  -100, False,  ), 
    (  (1, 1890), 'int64',  -100, False,  ), 
    (  (38), 'int64',  0.0, False,  ), 
    (  (1, 62), 'int64',  0, False,  ), 
    (  (1070), 'float32',  49408, False,  ), 
    (  (1035), 'float32',  -100, False,  ), 
    (  (1041), 'float32',  -100, False,  ), 
    (  (1, 67), 'int64',  -100, False,  ), 
    (  (1, 1061), 'int64',  0, False,  ), 
    (  (1, 17), 'int64',  -100, False,  ), 
    (  (1, 16), 'int64',  -100, False,  ), 
    (  (1890), 'int64',  0.0, False,  ), 
    (  (1, 1290), 'int64',  0, False,  ), 
    (  (1045), 'float32',  49408, False,  ), 
    (  (1, 55), 'int64',  0.0, False,  ), 
    (  (1, 1067), 'int64',  -100, False,  ), 
    (  (1063), 'int64',  0, False,  ), 
    (  (1, 25), 'int64',  -100, False,  ), 
    (  (1, 41), 'int64',  0, False,  ), 
    (  (1, 1881), 'int64',  0, False,  ), 
    (  (1, 2983), 'int64',  -100, False,  ), 
    (  (1, 1046), 'int64',  -100, False,  ), 
    (  (1, 15), 'int64',  0.0, False,  ), 
    (  (1, 13), 'int64',  0.0, False,  ), 
    (  (11), 'int64',  0.0, False,  ), 

}

class TestFillConstantDevelopCase3(unittest.TestCase):
    def cal_torch_res(self):
        out = torch.full(size=self.shape, fill_value=self.value, dtype=convert_dtype_to_torch_type(self.dtype))
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self):
        out = paddle.tensor.fill_constant(self.shape, self.dtype, self.value, self.force_cpu)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out
    
    def test_eager_accuracy(self):
        for (shape, dtype, value, force_cpu) in all_case:
            self.shape = shape
            self.dtype = dtype
            self.value = value
            self.force_cpu = force_cpu

            if not isinstance(self.shape, tuple):
                self.shape = (self.shape, )

            out_torch = self.cal_torch_res()
            self.out_torch = out_torch.cpu().detach().numpy()
            del out_torch
            torch.cuda.empty_cache()
            out_eager = self.cal_eager_res()
            out_eager_np = out_eager.numpy()
            del out_eager
            paddle.device.cuda.empty_cache()
            np.testing.assert_equal(out_eager_np, self.out_torch, err_msg=f'compare equal eager forward res with torch failed in shape: {self.shape}, dtype: {self.dtype}, value: {self.value}, force_cpu: {self.force_cpu}.')



if __name__ == '__main__':
    generate_np_inputs()
    unittest.main()
