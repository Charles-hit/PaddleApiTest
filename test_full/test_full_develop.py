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

if __name__ == '__main__':
    generate_np_inputs()
    unittest.main()
