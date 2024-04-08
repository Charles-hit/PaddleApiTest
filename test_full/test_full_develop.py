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


class TestFullDevelopCase1_FP32(unittest.TestCase):
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
        out = paddle.full(shape, fill_value, dtype)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def cal_static_res(self, shape, fill_value, dtype):
        out = paddle.full(shape, fill_value, dtype)
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
            api="full",
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
            api="full",
        )

    def test_eager_stability(self):
        shape_eager, fill_value_eager, dtype_eager = self.gen_eager_inputs()
        out_eager_baseline = self.cal_eager_res(
            shape_eager, fill_value_eager, dtype_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
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
                api="full",
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
            for i in range(5):
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
                    api="full",
                )


class TestFullDevelopCase1_FP16(TestFullDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"


class TestFullDevelopCase1_BF16(TestFullDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"


class TestFullDevelopCase2_FP32(TestFullDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp32.npz"


class TestFullDevelopCase2_FP16(TestFullDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp16.npz"


class TestFullDevelopCase2_BF16(TestFullDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npz"


class TestFullDevelopCase3_INT64(unittest.TestCase):
    def test_eager(self):
        shape = ()
        fill_value = 1
        eager_out = paddle.full(shape, 1, dtype="int64")
        torch_out = torch.full(size=shape, fill_value=1, dtype=torch.int64)
        self.assertEqual(eager_out, torch_out.cpu().detach().numpy())
        for i in range(5):
            eager_out = paddle.full(shape, 1, dtype="int64")
            torch_out = torch.full(size=shape, fill_value=1, dtype=torch.int64)
            self.assertEqual(eager_out, torch_out.cpu().detach().numpy())

    def test_static(self):
        paddle.enable_static()
        shape = ()
        fill_value = 1
        mp, sp = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            out_static = paddle.full(shape, 1, dtype="int64")
        exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
        exe.run(sp)
        out = exe.run(
            mp,
            fetch_list=[out_static],
        )
        out_static = out[0]
        torch_out = torch.full(size=shape, fill_value=1, dtype=torch.int64)
        self.assertEqual(out_static, torch_out.cpu().detach().numpy())
        for i in range(5):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                out_static = paddle.full(shape, 1, dtype="int64")
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                fetch_list=[out_static],
            )
            out_static = out[0]
            torch_out = torch.full(size=shape, fill_value=1, dtype=torch.int64)
            self.assertEqual(out_static, torch_out.cpu().detach().numpy())

# shape, fill_value, dtype
all_case = {
    # MP1
    (  (1, 74),  0, 'int64',  ), 
    (  (1, 1099),  -100, 'int64',  ), 
    (  (1, 13),  0, 'int64',  ), 
    (  (1, 1037),  0, 'int64',  ), 
    (  (1, 15),  0, 'int64',  ), 
    (  (1077),  49408, None,  ), 
    (  (1, 1064),  0, 'int64',  ), 
    (  (1, 1047),  -100, 'int64',  ), 
    (  (1, 1050),  0, 'int64',  ), 
    (  (1, 12),  -100, 'int64',  ), 
    (  (1, 14),  0, 'int64',  ), 
    (  (1, 1078),  -100, 'int64',  ), 
    (  (1052),  49408, None,  ), 
    (  (1, 1046),  0, 'int64',  ), 
    (  (1, 54),  0, 'int64',  ), 
    (  (1, 63),  0, 'int64',  ), 
    (  (1, 26),  0, 'int64',  ), 
    (  (1, 36),  0, 'int64',  ), 
    (  (1065),  -100, None,  ), 
    (  (1, 23),  0, 'int64',  ), 
    (  (1, 1058),  0, 'int64',  ), 
    (  (1379),  -100, None,  ), 
    (  (1, 58),  -100, 'int64',  ), 
    (  (1, 1050),  -100, 'int64',  ), 
    (  (1, 12),  0, 'int64',  ), 
    (  (1049),  -100, None,  ), 
    (  (1, 29),  -100, 'int64',  ), 
    (  (1, 1049),  0, 'int64',  ), 
    (  (1050),  49408, None,  ), 
    (  (1, 57),  0, 'int64',  ), 
    (  (1, 66),  0, 'int64',  ), 
    (  (1, 73),  0, 'int64',  ), 
    (  (1064),  49408, None,  ), 
    (  (1, 1099),  0, 'int64',  ), 
    (  (1, 74),  -100, 'int64',  ), 
    (  (1043),  49408, None,  ), 
    (  (1, 41),  0, 'int64',  ), 
    (  (1, 1066),  -100, 'int64',  ), 
    (  (1, 75),  -100, 'int64',  ), 
    (  (1051),  -100, None,  ), 
    (  (1, 2066),  0, 'int64',  ), 
    (  (1045),  49408, None,  ), 
    (  (1048),  -100, None,  ), 
    (  (1034),  49408, None,  ), 
    (  (1076),  -100, None,  ), 
    (  (1061),  -100, None,  ), 
    (  (1, 1065),  -100, 'int64',  ), 
    (  (1, 19),  -100, 'int64',  ), 
    (  (1066),  49408, None,  ), 
    (  (1, 33),  -100, 'int64',  ), 
    (  (1, 65),  -100, 'int64',  ), 
    (  (1, 51),  0, 'int64',  ), 
    (  (1, 52),  -100, 'int64',  ), 
    (  (1, 356),  -100, 'int64',  ), 
    (  (1058),  49408, None,  ), 
    (  (1, 1045),  -100, 'int64',  ), 
    (  (1059),  49408, None,  ), 
    (  (1064),  -100, None,  ), 
    (  (1, 2066),  -100, 'int64',  ), 
    (  (1, 1034),  -100, 'int64',  ), 
    (  (1, 31),  -100, 'int64',  ), 
    (  (1, 1066),  0, 'int64',  ), 
    (  (1046),  -100, None,  ), 
    (  (1, 1041),  0, 'int64',  ), 
    (  (1, 1039),  -100, 'int64',  ), 
    (  (1, 1061),  0, 'int64',  ), 
    (  (1, 38),  0, 'int64',  ), 
    (  (1051),  49408, None,  ), 
    (  (1044),  -100, None,  ), 
    (  (1, 30),  0, 'int64',  ), 
    (  (1, 47),  0, 'int64',  ), 
    (  (1, 46),  -100, 'int64',  ), 
    (  (1, 17),  -100, 'int64',  ), 
    (  (1, 1041),  -100, 'int64',  ), 
    (  (1, 1049),  -100, 'int64',  ), 
    (  (1, 1052),  0, 'int64',  ), 
    (  (1052),  -100, None,  ), 
    (  (1, 47),  -100, 'int64',  ), 
    (  (1, 1052),  -100, 'int64',  ), 
    (  (1, 1044),  0, 'int64',  ), 
    (  (1, 20),  -100, 'int64',  ), 
    (  (1, 57),  -100, 'int64',  ), 
    (  (1, 53),  0, 'int64',  ), 
    (  (1, 1090),  -100, 'int64',  ), 
    (  (1035),  -100, None,  ), 
    (  (1039),  49408, None,  ), 
    (  (1, 18),  -100, 'int64',  ), 
    (  (1061),  49408, None,  ), 
    (  (1, 1058),  -100, 'int64',  ), 
    (  (1, 26),  -100, 'int64',  ), 
    (  (1, 62),  -100, 'int64',  ), 
    (  (1035),  49408, None,  ), 
    (  (1040),  49408, None,  ), 
    (  (1055),  -100, None,  ), 
    (  (1, 1056),  0, 'int64',  ), 
    (  (1055),  49408, None,  ), 
    (  (1, 40),  0, 'int64',  ), 
    (  (1090),  49408, None,  ), 
    (  (1, 76),  -100, 'int64',  ), 
    (  (1, 32),  0, 'int64',  ), 
    (  (1067),  49408, None,  ), 
    (  (1, 1077),  0, 'int64',  ), 
    (  (1058),  -100, None,  ), 
    (  (1, 49),  -100, 'int64',  ), 
    (  (1, 10),  0, 'int64',  ), 
    (  (1, 9),  0, 'int64',  ), 
    (  (1, 28),  0, 'int64',  ), 
    (  (1, 3),  0, 'int64',  ), 
    (  (1078),  -100, None,  ), 
    (  (1069),  -100, None,  ), 
    (  (1, 18),  0, 'int64',  ), 
    (  (1, 1036),  0, 'int64',  ), 
    (  (1, 1048),  -100, 'int64',  ), 
    (  (1050),  -100, None,  ), 
    (  (1, 1077),  -100, 'int64',  ), 
    (  (1, 64),  0, 'int64',  ), 
    (  (1, 45),  0, 'int64',  ), 
    (  (1, 34),  -100, 'int64',  ), 
    (  (1054),  -100, None,  ), 
    (  (1065),  49408, None,  ), 
    (  (1045),  -100, None,  ), 
    (  (1, 1054),  -100, 'int64',  ), 
    (  (1056),  -100, None,  ), 
    (  (1, 1053),  0, 'int64',  ), 
    (  (1053),  49408, None,  ), 
    (  (1, 1054),  0, 'int64',  ), 
    (  (1, 19),  0, 'int64',  ), 
    (  (1, 1039),  0, 'int64',  ), 
    (  (1, 1067),  -100, 'int64',  ), 
    (  (1, 39),  0, 'int64',  ), 
    (  (1, 1065),  0, 'int64',  ), 
    (  (1, 43),  0, 'int64',  ), 
    (  (1, 35),  0, 'int64',  ), 
    (  (1, 1042),  -100, 'int64',  ), 
    (  (1041),  -100, None,  ), 
    (  (1, 24),  -100, 'int64',  ), 
    (  (1090),  -100, None,  ), 
    (  (1, 1057),  0, 'int64',  ), 
    (  (1036),  -100, None,  ), 
    (  (1, 1064),  -100, 'int64',  ), 
    (  (1099),  -100, None,  ), 
    (  (1077),  -100, None,  ), 
    (  (1038),  49408, None,  ), 
    (  (1, 66),  -100, 'int64',  ), 
    (  (1, 32),  -100, 'int64',  ), 
    (  (1, 63),  -100, 'int64',  ), 
    (  (1057),  -100, None,  ), 
    (  (1, 24),  0, 'int64',  ), 
    (  (1, 75),  0, 'int64',  ), 
    (  (1, 13),  -100, 'int64',  ), 
    (  (1, 31),  0, 'int64',  ), 
    (  (1048),  49408, None,  ), 
    (  (1, 356),  0, 'int64',  ), 
    (  (1, 1078),  0, 'int64',  ), 
    (  (1057),  49408, None,  ), 
    (  (1099),  49408, None,  ), 
    (  (1, 1045),  0, 'int64',  ), 
    (  (1069),  49408, None,  ), 
    (  (1, 1090),  0, 'int64',  ), 
    (  (1, 17),  0, 'int64',  ), 
    (  (1, 1035),  0, 'int64',  ), 
    (  (1, 1061),  -100, 'int64',  ), 
    (  (1044),  49408, None,  ), 
    (  (1, 1051),  -100, 'int64',  ), 
    (  (1, 65),  0, 'int64',  ), 
    (  (1, 1043),  -100, 'int64',  ), 
    (  (1, 1067),  0, 'int64',  ), 
    (  (1, 1379),  0, 'int64',  ), 
    (  (1, 1076),  0, 'int64',  ), 
    (  (1, 39),  -100, 'int64',  ), 
    (  (1, 28),  -100, 'int64',  ), 
    (  (1062),  -100, None,  ), 
    (  (1, 68),  -100, 'int64',  ), 
    (  (1, 71),  -100, 'int64',  ), 
    (  (1039),  -100, None,  ), 
    (  (1, 35),  -100, 'int64',  ), 
    (  (1, 1062),  -100, 'int64',  ), 
    (  (1054),  49408, None,  ), 
    (  (1, 16),  0, 'int64',  ), 
    (  (1, 1035),  -100, 'int64',  ), 
    (  (1, 1056),  -100, 'int64',  ), 
    (  (1, 1051),  0, 'int64',  ), 
    (  (1047),  -100, None,  ), 
    (  (1038),  -100, None,  ), 
    (  (1042),  -100, None,  ), 
    (  (1, 72),  0, 'int64',  ), 
    (  (1, 1043),  0, 'int64',  ), 
    (  (1, 1055),  0, 'int64',  ), 
    (  (1, 1076),  -100, 'int64',  ), 
    (  (1, 1071),  0, 'int64',  ), 
    (  (1034),  -100, None,  ), 
    (  (1046),  49408, None,  ), 
    (  (1, 55),  -100, 'int64',  ), 
    (  (1, 1059),  0, 'int64',  ), 
    (  (1059),  -100, None,  ), 
    (  (1, 1046),  -100, 'int64',  ), 
    (  (1, 1038),  0, 'int64',  ), 
    (  (1, 34),  0, 'int64',  ), 
    (  (1078),  49408, None,  ), 
    (  (1, 11),  -100, 'int64',  ), 
    (  (1, 48),  -100, 'int64',  ), 
    (  (1, 15),  -100, 'int64',  ), 
    (  (1, 33),  0, 'int64',  ), 
    (  (1, 25),  0, 'int64',  ), 
    (  (1, 43),  -100, 'int64',  ), 
    (  (1, 1059),  -100, 'int64',  ), 
    (  (1037),  -100, None,  ), 
    (  (1, 1037),  -100, 'int64',  ), 
    (  (1071),  49408, None,  ), 
    (  (1, 72),  -100, 'int64',  ), 
    (  (1, 70),  0, 'int64',  ), 
    (  (1, 37),  0, 'int64',  ), 
    (  (1, 8),  0, 'int64',  ), 
    (  (1062),  49408, None,  ), 
    (  (1, 29),  0, 'int64',  ), 
    (  (1053),  -100, None,  ), 
    (  (1066),  -100, None,  ), 
    (  (1, 1062),  0, 'int64',  ), 
    (  (1042),  49408, None,  ), 
    (  (1, 69),  -100, 'int64',  ), 
    (  (1067),  -100, None,  ), 
    (  (1, 1047),  0, 'int64',  ), 
    (  (1, 42),  0, 'int64',  ), 
    (  (1, 6),  0, 'int64',  ), 
    (  (1, 73),  -100, 'int64',  ), 
    (  (1, 14),  -100, 'int64',  ), 
    (  (1, 21),  0, 'int64',  ), 
    (  (1, 55),  0, 'int64',  ), 
    (  (1, 68),  0, 'int64',  ), 
    (  (1, 22),  -100, 'int64',  ), 
    (  (1, 11),  0, 'int64',  ), 
    (  (1, 20),  0, 'int64',  ), 
    (  (1, 41),  -100, 'int64',  ), 
    (  (1, 1040),  0, 'int64',  ), 
    (  (1379),  49408, None,  ), 
    (  (1, 1071),  -100, 'int64',  ), 
    (  (1, 51),  -100, 'int64',  ), 
    (  (1, 21),  -100, 'int64',  ), 
    (  (1, 53),  -100, 'int64',  ), 
    (  (1041),  49408, None,  ), 
    (  (1, 62),  0, 'int64',  ), 
    (  (1, 1),  1024, 'int64',  ), 
    (  (1, 27),  0, 'int64',  ), 
    (  (1, 64),  -100, 'int64',  ), 
    (  (1, 16),  -100, 'int64',  ), 
    (  (1047),  49408, None,  ), 
    (  (1, 25),  -100, 'int64',  ), 
    (  (1, 36),  -100, 'int64',  ), 
    (  (1036),  49408, None,  ), 
    (  (1, 58),  0, 'int64',  ), 
    (  (1, 5),  0, 'int64',  ), 
    (  (1, 1040),  -100, 'int64',  ), 
    (  (1, 1042),  0, 'int64',  ), 
    (  (1, 1036),  -100, 'int64',  ), 
    (  (1, 49),  0, 'int64',  ), 
    (  (1, 1053),  -100, 'int64',  ), 
    (  (1, 48),  0, 'int64',  ), 
    (  (1, 1034),  0, 'int64',  ), 
    (  (1, 42),  -100, 'int64',  ), 
    (  (1076),  49408, None,  ), 
    (  (1, 1069),  -100, 'int64',  ), 
    (  (1, 69),  0, 'int64',  ), 
    (  (1, 1044),  -100, 'int64',  ), 
    (  (1, 1055),  -100, 'int64',  ), 
    (  (1, 1069),  0, 'int64',  ), 
    (  (1, 59),  0, 'int64',  ), 
    (  (1, 30),  -100, 'int64',  ), 
    (  (1056),  49408, None,  ), 
    (  (1, 1379),  -100, 'int64',  ), 
    (  (1, 76),  0, 'int64',  ), 
    (  (1049),  49408, None,  ), 
    (  (1071),  -100, None,  ), 
    (  (1040),  -100, None,  ), 
    (  (1, 71),  0, 'int64',  ), 
    (  (1, 38),  -100, 'int64',  ), 
    (  (1, 22),  0, 'int64',  ), 
    (  (1, 54),  -100, 'int64',  ), 
    (  (1, 67),  -100, 'int64',  ), 
    (  (1, 1038),  -100, 'int64',  ), 
    (  (1, 7),  0, 'int64',  ), 
    (  (1, 67),  0, 'int64',  ), 
    (  (1, 52),  0, 'int64',  ), 
    (  (1, 4),  0, 'int64',  ), 
    (  (1, 44),  -100, 'int64',  ), 
    (  (1, 1048),  0, 'int64',  ), 
    (  (1, 1057),  -100, 'int64',  ), 
    (  (1043),  -100, None,  ), 
    (  (1),  5.0, 'float32',  ), 
    (  (1, 46),  0, 'int64',  ), 
    (  (1, 70),  -100, 'int64',  ), 
    (  (1, 44),  0, 'int64',  ), 
    (  (1037),  49408, None,  ), 
    (  (1, 23),  -100, 'int64',  ), 
    (  (1, 27),  -100, 'int64',  ), 
    # MP8
    (  (1071),  49408, None,  ), 
    (  (1, 1815),  0, 'int64',  ), 
    (  (1072),  49408, None,  ), 
    (  (1, 1057),  -100, 'int64',  ), 
    (  (1, 34),  0, 'int64',  ), 
    (  (1, 1040),  -100, 'int64',  ), 
    (  (1, 1056),  0, 'int64',  ), 
    (  (1, 35),  -100, 'int64',  ), 
    (  (1, 1043),  0, 'int64',  ), 
    (  (1051),  49408, None,  ), 
    (  (1, 1060),  0, 'int64',  ), 
    (  (1, 5),  0, 'int64',  ), 
    (  (1, 57),  0, 'int64',  ), 
    (  (1063),  -100, None,  ), 
    (  (1, 68),  0, 'int64',  ), 
    (  (1, 20),  0, 'int64',  ), 
    (  (1, 1722),  -100, 'int64',  ), 
    (  (1, 1561),  0, 'int64',  ), 
    (  (1, 15),  0, 'int64',  ), 
    (  (1, 1049),  0, 'int64',  ), 
    (  (1, 2339),  0, 'int64',  ), 
    (  (1, 56),  -100, 'int64',  ), 
    (  (1, 1072),  0, 'int64',  ), 
    (  (1, 3041),  -100, 'int64',  ), 
    (  (1, 26),  0, 'int64',  ), 
    (  (1, 16),  0, 'int64',  ), 
    (  (1, 1052),  -100, 'int64',  ), 
    (  (1),  5.0, 'float32',  ), 
    (  (1, 16),  -100, 'int64',  ), 
    (  (1, 1041),  -100, 'int64',  ), 
    (  (1, 18),  0, 'int64',  ), 
    (  (1, 1316),  0, 'int64',  ), 
    (  (2339),  49408, None,  ), 
    (  (2339),  -100, None,  ), 
    (  (1, 19),  0, 'int64',  ), 
    (  (1, 21),  -100, 'int64',  ), 
    (  (1, 1960),  -100, 'int64',  ), 
    (  (1, 1078),  0, 'int64',  ), 
    (  (1, 1298),  0, 'int64',  ), 
    (  (1, 1066),  0, 'int64',  ), 
    (  (1038),  49408, None,  ), 
    (  (1, 1034),  0, 'int64',  ), 
    (  (1, 12),  -100, 'int64',  ), 
    (  (1046),  -100, None,  ), 
    (  (1, 1035),  -100, 'int64',  ), 
    (  (1, 1290),  -100, 'int64',  ), 
    (  (1042),  -100, None,  ), 
    (  (1058),  49408, None,  ), 
    (  (1061),  -100, None,  ), 
    (  (1, 61),  -100, 'int64',  ), 
    (  (1, 1061),  0, 'int64',  ), 
    (  (1, 2955),  -100, 'int64',  ), 
    (  (1051),  -100, None,  ), 
    (  (1, 2287),  0, 'int64',  ), 
    (  (1, 1058),  0, 'int64',  ), 
    (  (1, 47),  -100, 'int64',  ), 
    (  (1, 1052),  0, 'int64',  ), 
    (  (1046),  49408, None,  ), 
    (  (1045),  49408, None,  ), 
    (  (1, 49),  -100, 'int64',  ), 
    (  (1060),  -100, None,  ), 
    (  (1077),  -100, None,  ), 
    (  (1, 2913),  0, 'int64',  ), 
    (  (1, 1081),  -100, 'int64',  ), 
    (  (1, 7),  0, 'int64',  ), 
    (  (1, 1070),  0, 'int64',  ), 
    (  (1, 1049),  -100, 'int64',  ), 
    (  (1, 1045),  0, 'int64',  ), 
    (  (1048),  49408, None,  ), 
    (  (1, 31),  -100, 'int64',  ), 
    (  (1038),  -100, None,  ), 
    (  (1, 56),  0, 'int64',  ), 
    (  (1050),  49408, None,  ), 
    (  (1, 2983),  -100, 'int64',  ), 
    (  (1037),  49408, None,  ), 
    (  (1, 50),  0, 'int64',  ), 
    (  (1, 32),  0, 'int64',  ), 
    (  (1, 1038),  -100, 'int64',  ), 
    (  (2745),  -100, None,  ), 
    (  (1, 3041),  0, 'int64',  ), 
    (  (1060),  49408, None,  ), 
    (  (1, 1048),  -100, 'int64',  ), 
    (  (1, 58),  -100, 'int64',  ), 
    (  (1, 27),  -100, 'int64',  ), 
    (  (1047),  -100, None,  ), 
    (  (1048),  -100, None,  ), 
    (  (1070),  -100, None,  ), 
    (  (1035),  49408, None,  ), 
    (  (1, 42),  -100, 'int64',  ), 
    (  (2955),  49408, None,  ), 
    (  (1, 1042),  0, 'int64',  ), 
    (  (1, 1890),  0, 'int64',  ), 
    (  (1059),  49408, None,  ), 
    (  (1, 38),  0, 'int64',  ), 
    (  (1062),  -100, None,  ), 
    (  (1052),  49408, None,  ), 
    (  (1043),  49408, None,  ), 
    (  (2913),  -100, None,  ), 
    (  (1050),  -100, None,  ), 
    (  (1, 1055),  -100, 'int64',  ), 
    (  (1, 1056),  -100, 'int64',  ), 
    (  (1, 8),  0, 'int64',  ), 
    (  (1, 37),  0, 'int64',  ), 
    (  (1034),  49408, None,  ), 
    (  (1, 1040),  0, 'int64',  ), 
    (  (1057),  -100, None,  ), 
    (  (1056),  49408, None,  ), 
    (  (1054),  -100, None,  ), 
    (  (2745),  49408, None,  ), 
    (  (1, 1037),  -100, 'int64',  ), 
    (  (1, 26),  -100, 'int64',  ), 
    (  (1043),  -100, None,  ), 
    (  (1, 41),  -100, 'int64',  ), 
    (  (1072),  -100, None,  ), 
    (  (1, 1044),  0, 'int64',  ), 
    (  (1, 1345),  -100, 'int64',  ), 
    (  (1, 47),  0, 'int64',  ), 
    (  (1, 20),  -100, 'int64',  ), 
    (  (1, 66),  0, 'int64',  ), 
    (  (1, 1039),  -100, 'int64',  ), 
    (  (1, 51),  0, 'int64',  ), 
    (  (1063),  49408, None,  ), 
    (  (1, 1039),  0, 'int64',  ), 
    (  (1, 43),  0, 'int64',  ), 
    (  (1, 60),  0, 'int64',  ), 
    (  (1, 17),  -100, 'int64',  ), 
    (  (1, 1264),  -100, 'int64',  ), 
    (  (1, 1),  1024, 'int64',  ), 
    (  (1, 1053),  0, 'int64',  ), 
    (  (1, 62),  0, 'int64',  ), 
    (  (1070),  49408, None,  ), 
    (  (1, 1051),  0, 'int64',  ), 
    (  (1, 1043),  -100, 'int64',  ), 
    (  (1, 1036),  -100, 'int64',  ), 
    (  (1, 57),  -100, 'int64',  ), 
    (  (1, 73),  0, 'int64',  ), 
    (  (1078),  49408, None,  ), 
    (  (1049),  -100, None,  ), 
    (  (1, 23),  -100, 'int64',  ), 
    (  (1, 45),  0, 'int64',  ), 
    (  (1, 2018),  0, 'int64',  ), 
    (  (1, 48),  0, 'int64',  ), 
    (  (1, 2983),  0, 'int64',  ), 
    (  (1, 1071),  0, 'int64',  ), 
    (  (1, 1881),  0, 'int64',  ), 
    (  (1, 21),  0, 'int64',  ), 
    (  (1, 9),  0, 'int64',  ), 
    (  (1, 1045),  -100, 'int64',  ), 
    (  (1, 50),  -100, 'int64',  ), 
    (  (2287),  -100, None,  ), 
    (  (1, 65),  0, 'int64',  ), 
    (  (1, 2955),  0, 'int64',  ), 
    (  (1, 1737),  -100, 'int64',  ), 
    (  (1, 1932),  -100, 'int64',  ), 
    (  (1, 10),  0, 'int64',  ), 
    (  (1, 6),  0, 'int64',  ), 
    (  (1078),  -100, None,  ), 
    (  (1, 1059),  -100, 'int64',  ), 
    (  (1034),  -100, None,  ), 
    (  (1036),  -100, None,  ), 
    (  (1, 23),  0, 'int64',  ), 
    (  (1036),  49408, None,  ), 
    (  (1, 17),  0, 'int64',  ), 
    (  (1, 1035),  0, 'int64',  ), 
    (  (1, 1034),  -100, 'int64',  ), 
    (  (1066),  49408, None,  ), 
    (  (1, 29),  -100, 'int64',  ), 
    (  (1, 1067),  0, 'int64',  ), 
    (  (1, 40),  -100, 'int64',  ), 
    (  (1052),  -100, None,  ), 
    (  (1, 40),  0, 'int64',  ), 
    (  (2377),  -100, None,  ), 
    (  (1, 1354),  -100, 'int64',  ), 
    (  (1, 1354),  0, 'int64',  ), 
    (  (1, 1057),  0, 'int64',  ), 
    (  (1, 11),  -100, 'int64',  ), 
    (  (1, 46),  -100, 'int64',  ), 
    (  (1, 2018),  -100, 'int64',  ), 
    (  (1, 44),  -100, 'int64',  ), 
    (  (1081),  -100, None,  ), 
    (  (1, 14),  0, 'int64',  ), 
    (  (1077),  49408, None,  ), 
    (  (1, 45),  -100, 'int64',  ), 
    (  (1, 46),  0, 'int64',  ), 
    (  (1, 66),  -100, 'int64',  ), 
    (  (1, 2377),  0, 'int64',  ), 
    (  (1, 1722),  0, 'int64',  ), 
    (  (1039),  49408, None,  ), 
    (  (1, 55),  0, 'int64',  ), 
    (  (1, 1061),  -100, 'int64',  ), 
    (  (1, 2339),  -100, 'int64',  ), 
    (  (1, 33),  0, 'int64',  ), 
    (  (1, 14),  -100, 'int64',  ), 
    (  (1, 1561),  -100, 'int64',  ), 
    (  (1, 1078),  -100, 'int64',  ), 
    (  (1, 76),  0, 'int64',  ), 
    (  (1, 1037),  0, 'int64',  ), 
    (  (1, 24),  0, 'int64',  ), 
    (  (1035),  -100, None,  ), 
    (  (1, 15),  -100, 'int64',  ), 
    (  (1, 12),  0, 'int64',  ), 
    (  (1, 1055),  0, 'int64',  ), 
    (  (1, 1059),  0, 'int64',  ), 
    (  (1049),  49408, None,  ), 
    (  (1, 1890),  -100, 'int64',  ), 
    (  (1, 55),  -100, 'int64',  ), 
    (  (1, 42),  0, 'int64',  ), 
    (  (1, 1071),  -100, 'int64',  ), 
    (  (1, 1345),  0, 'int64',  ), 
    (  (1, 1053),  -100, 'int64',  ), 
    (  (1067),  -100, None,  ), 
    (  (1, 1046),  -100, 'int64',  ), 
    (  (1055),  -100, None,  ), 
    (  (2955),  -100, None,  ), 
    (  (1, 30),  -100, 'int64',  ), 
    (  (1, 1046),  0, 'int64',  ), 
    (  (2983),  49408, None,  ), 
    (  (1, 1420),  0, 'int64',  ), 
    (  (1, 28),  -100, 'int64',  ), 
    (  (1, 24),  -100, 'int64',  ), 
    (  (1058),  -100, None,  ), 
    (  (1, 1044),  -100, 'int64',  ), 
    (  (1, 11),  0, 'int64',  ), 
    (  (1044),  -100, None,  ), 
    (  (3041),  49408, None,  ), 
    (  (1, 1058),  -100, 'int64',  ), 
    (  (1, 1077),  -100, 'int64',  ), 
    (  (2377),  49408, None,  ), 
    (  (1, 22),  0, 'int64',  ), 
    (  (1, 48),  -100, 'int64',  ), 
    (  (1, 22),  -100, 'int64',  ), 
    (  (1, 25),  0, 'int64',  ), 
    (  (1, 62),  -100, 'int64',  ), 
    (  (1, 34),  -100, 'int64',  ), 
    (  (1, 29),  0, 'int64',  ), 
    (  (1, 1737),  0, 'int64',  ), 
    (  (1, 2287),  -100, 'int64',  ), 
    (  (1, 53),  -100, 'int64',  ), 
    (  (1, 39),  -100, 'int64',  ), 
    (  (1, 1420),  -100, 'int64',  ), 
    (  (1, 1063),  0, 'int64',  ), 
    (  (1, 59),  0, 'int64',  ), 
    (  (1, 39),  0, 'int64',  ), 
    (  (1, 1062),  0, 'int64',  ), 
    (  (2983),  -100, None,  ), 
    (  (1071),  -100, None,  ), 
    (  (1, 76),  -100, 'int64',  ), 
    (  (1, 54),  0, 'int64',  ), 
    (  (1, 1062),  -100, 'int64',  ), 
    (  (1, 49),  0, 'int64',  ), 
    (  (1, 1081),  0, 'int64',  ), 
    (  (1, 1070),  -100, 'int64',  ), 
    (  (1056),  -100, None,  ), 
    (  (1041),  49408, None,  ), 
    (  (1, 1066),  -100, 'int64',  ), 
    (  (1, 1316),  -100, 'int64',  ), 
    (  (1, 1298),  -100, 'int64',  ), 
    (  (1, 2745),  0, 'int64',  ), 
    (  (1055),  49408, None,  ), 
    (  (1, 73),  -100, 'int64',  ), 
    (  (1, 18),  -100, 'int64',  ), 
    (  (1, 36),  -100, 'int64',  ), 
    (  (1061),  49408, None,  ), 
    (  (1, 3),  0, 'int64',  ), 
    (  (3041),  -100, None,  ), 
    (  (1, 1054),  -100, 'int64',  ), 
    (  (1, 2745),  -100, 'int64',  ), 
    (  (1, 61),  0, 'int64',  ), 
    (  (1, 67),  -100, 'int64',  ), 
    (  (1, 52),  0, 'int64',  ), 
    (  (1081),  49408, None,  ), 
    (  (1, 1815),  -100, 'int64',  ), 
    (  (1, 2377),  -100, 'int64',  ), 
    (  (1039),  -100, None,  ), 
    (  (2287),  49408, None,  ), 
    (  (1, 28),  0, 'int64',  ), 
    (  (1, 52),  -100, 'int64',  ), 
    (  (1, 1041),  0, 'int64',  ), 
    (  (1059),  -100, None,  ), 
    (  (1, 44),  0, 'int64',  ), 
    (  (1, 1048),  0, 'int64',  ), 
    (  (1, 19),  -100, 'int64',  ), 
    (  (1, 59),  -100, 'int64',  ), 
    (  (1, 2913),  -100, 'int64',  ), 
    (  (1, 36),  0, 'int64',  ), 
    (  (1, 60),  -100, 'int64',  ), 
    (  (1053),  -100, None,  ), 
    (  (1, 1047),  0, 'int64',  ), 
    (  (1, 67),  0, 'int64',  ), 
    (  (1, 1060),  -100, 'int64',  ), 
    (  (1, 1042),  -100, 'int64',  ), 
    (  (1062),  49408, None,  ), 
    (  (1054),  49408, None,  ), 
    (  (1, 41),  0, 'int64',  ), 
    (  (1, 1050),  -100, 'int64',  ), 
    (  (1, 13),  -100, 'int64',  ), 
    (  (1040),  -100, None,  ), 
    (  (1, 54),  -100, 'int64',  ), 
    (  (1042),  49408, None,  ), 
    (  (1, 43),  -100, 'int64',  ), 
    (  (1, 58),  0, 'int64',  ), 
    (  (1, 1067),  -100, 'int64',  ), 
    (  (1, 27),  0, 'int64',  ), 
    (  (1, 1881),  -100, 'int64',  ), 
    (  (1, 25),  -100, 'int64',  ), 
    (  (1066),  -100, None,  ), 
    (  (1, 1036),  0, 'int64',  ), 
    (  (1, 38),  -100, 'int64',  ), 
    (  (1, 51),  -100, 'int64',  ), 
    (  (1, 35),  0, 'int64',  ), 
    (  (1045),  -100, None,  ), 
    (  (1, 1050),  0, 'int64',  ), 
    (  (1044),  49408, None,  ), 
    (  (1, 30),  0, 'int64',  ), 
    (  (1, 1063),  -100, 'int64',  ), 
    (  (1, 1047),  -100, 'int64',  ), 
    (  (1, 32),  -100, 'int64',  ), 
    (  (1040),  49408, None,  ), 
    (  (1, 37),  -100, 'int64',  ), 
    (  (1, 1077),  0, 'int64',  ), 
    (  (1053),  49408, None,  ), 
    (  (1, 1290),  0, 'int64',  ), 
    (  (1, 1054),  0, 'int64',  ), 
    (  (1, 53),  0, 'int64',  ), 
    (  (1, 1932),  0, 'int64',  ), 
    (  (1, 13),  0, 'int64',  ), 
    (  (1057),  49408, None,  ), 
    (  (1, 1072),  -100, 'int64',  ), 
    (  (1, 33),  -100, 'int64',  ), 
    (  (1, 1051),  -100, 'int64',  ), 
    (  (1037),  -100, None,  ), 
    (  (1, 4),  0, 'int64',  ), 
    (  (1, 1038),  0, 'int64',  ), 
    (  (1041),  -100, None,  ), 
    (  (2913),  49408, None,  ), 
    (  (1, 31),  0, 'int64',  ), 
    (  (1047),  49408, None,  ), 
    (  (1067),  49408, None,  ), 
    (  (1, 1264),  0, 'int64',  ), 
    (  (1, 1960),  0, 'int64',  ), 
}

class TestFullDevelopCase4(unittest.TestCase):
    def cal_torch_res(self):
        out = torch.full(size=self.shape, fill_value=self.fill_value, dtype=convert_dtype_to_torch_type(self.dtype) if self.dtype is not None else None)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self):
        out = paddle.full(self.shape, self.fill_value, self.dtype)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        for (shape, fill_value, dtype) in all_case:
            self.shape = shape
            self.fill_value = fill_value
            self.dtype = dtype

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
            np.testing.assert_equal(out_eager_np, self.out_torch, err_msg=f'compare equal eager forward res with torch failed in shape: {self.shape}, fill_value: {self.fill_value}, dtype: {self.dtype}.')


if __name__ == '__main__':
    generate_np_inputs()
    unittest.main()
