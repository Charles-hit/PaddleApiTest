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


def generate_np_inputs_and_dout():
    x_case1 = np.random.random(size=[1, 32, 4096, 192]).astype("float32") - 0.5
    y_case1 = np.random.random(size=[1, 32, 4096, 192]).astype("float32") - 0.5
    dout_case1 = (
        np.random.random(size=[1, 32, 4096, 4096]).astype("float32") - 0.5
    )

    x_case2 = np.random.random(size=[1, 32, 4096, 4096]).astype("float32") - 0.5
    y_case2 = np.random.random(size=[1, 32, 4096, 192]).astype("float32") - 0.5
    dout_case2 = (
        np.random.random(size=[1, 32, 4096, 192]).astype("float32") - 0.5
    )

    x_case3 = np.random.random(size=[4096, 1, 10944]).astype("float32") - 0.5
    y_case3 = np.random.random(size=[4096, 10944]).astype("float32") - 0.5
    dout_case3 = (
        np.random.random(size=[4096, 1, 4096]).astype("float32") - 0.5
    )

    x_case4 = np.random.random(size=[4096, 1, 2048]).astype("float32") - 0.5
    y_case4 = np.random.random(size=[2048, 4096]).astype("float32") - 0.5
    dout_case4 = (
        np.random.random(size=[4096, 1, 4096]).astype("float32") - 0.5
    )

    x_case5 = np.random.random(size=[4096, 1, 4096]).astype("float32") - 0.5
    y_case5 = np.random.random(size=[2048, 4096]).astype("float32") - 0.5
    dout_case5 = (
        np.random.random(size=[4096, 1, 2048]).astype("float32") - 0.5
    )

    x_case6 = np.random.random(size=[4096, 1, 4096]).astype("float32") - 0.5
    y_case6 = np.random.random(size=[5472, 4096]).astype("float32") - 0.5
    dout_case6 = (
        np.random.random(size=[4096, 1, 5472]).astype("float32") - 0.5
    )

    x_case7 = np.random.random(size=[4096, 1, 5472]).astype("float32") - 0.5
    y_case7 = np.random.random(size=[5472, 4096]).astype("float32") - 0.5
    dout_case7 = (
        np.random.random(size=[4096, 1, 4096]).astype("float32") - 0.5
    )

    x_case8 = np.random.random(size=[4096, 1, 6144]).astype("float32") - 0.5
    y_case8 = np.random.random(size=[4096, 6144]).astype("float32") - 0.5
    dout_case8 = (
        np.random.random(size=[4096, 1, 4096]).astype("float32") - 0.5
    )

    x_case9 = np.random.random(size=[4096, 1]).astype("float32") - 0.5
    y_case9 = np.random.random(size=[1, 64]).astype("float32") - 0.5
    dout_case9 = (
        np.random.random(size=[4096, 64]).astype("float32") - 0.5
    )

    x_case10 = np.random.random(size=[1, 8192, 14336]).astype("float32") - 0.5
    y_case10 = np.random.random(size=[14336, 12528]).astype("float32") - 0.5
    dout_case10 = (
        np.random.random(size=[1, 8192, 12528]).astype("float32") - 0.5
    )  

    x_case11 = np.random.random(size=[1, 8192, 14336]).astype("float32") - 0.5
    y_case11 = np.random.random(size=[14336, 5376]).astype("float32") - 0.5
    dout_case11 = (
        np.random.random(size=[1, 8192, 5376]).astype("float32") - 0.5
    )

    x_case12 = np.random.random(size=[1, 8192, 14336]).astype("float32")
    y_case12 = np.random.random(size=[14336, 5376]).astype("float32")
    b_case12 = np.random.random(size=[5376]).astype("float32")
    dout_case12 = np.random.random(size=[1, 8192, 5376]).astype("float32")

    np.savez("./inputs_case1.npz", x=x_case1, y=y_case1, dout=dout_case1)
    np.savez("./inputs_case2.npz", x=x_case2, y=y_case2, dout=dout_case2)
    np.savez("./inputs_case3.npz", x=x_case3, y=y_case3, dout=dout_case3)
    np.savez("./inputs_case4.npz", x=x_case4, y=y_case4, dout=dout_case4)
    np.savez("./inputs_case5.npz", x=x_case5, y=y_case5, dout=dout_case5)
    np.savez("./inputs_case6.npz", x=x_case6, y=y_case6, dout=dout_case6)
    np.savez("./inputs_case7.npz", x=x_case7, y=y_case7, dout=dout_case7)
    np.savez("./inputs_case8.npz", x=x_case8, y=y_case8, dout=dout_case8)
    np.savez("./inputs_case9.npz", x=x_case9, y=y_case9, dout=dout_case9)
    np.savez("./inputs_case10.npz", x=x_case10, y=y_case10, dout=dout_case10)
    np.savez("./inputs_case11.npz", x=x_case11, y=y_case11, dout=dout_case11)
    np.savez("./inputs_case12.npz", x=x_case12, y=y_case12, b=b_case12, dout=dout_case12)

class TestMatmulDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, y_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, y_torch, self.transpose_x, self.transpose_y, dout_torch
        )
        del x_torch
        del y_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp32.npz"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_x = np_inputs_array["x"]
        self.np_y = np_inputs_array["y"]
        self.np_dout = np_inputs_array["dout"]
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

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        y_static = paddle.static.data(
            'y',
            shape=self.np_y.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        y_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, y_static, dout_static

    def cal_torch_res(self, x, y, transpose_x, transpose_y, dout):
        x_t = x
        y_t = y
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            y_t = y.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        if transpose_x:
            x_t = torch.transpose(x_t, -1, -2)
        if transpose_y:
            y_t = torch.transpose(y_t, -1, -2)
        out = torch.matmul(x_t, y_t)
        out_grads = torch.autograd.grad([out], [x, y], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, x, y, transpose_x, transpose_y, dout):
        x_t = x
        y_t = y
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            y_t = paddle.cast(y, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.matmul(x_t, y_t, transpose_x, transpose_y)
        out_grads = paddle.grad([out], [x, y], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def cal_static_res(self, x, y, transpose_x, transpose_y, dout):
        x_t = x
        y_t = y
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            y_t = paddle.cast(y, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.matmul(x_t, y_t, transpose_x, transpose_y)
        out_grads = paddle.static.gradients(
            [out], [x, y], target_gradients=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, y_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, y_eager, self.transpose_x, self.transpose_y, dout_eager
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
        # save eager res for test_matmul_incubate
        np.savez(
            self.save_eager_res_path,
            out_eager=out_eager_np,
            out_grads_eager_0=out_grads_eager_np[0],
            out_grads_eager_1=out_grads_eager_np[1],
        )
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
            api="paddle.matmul",
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
                api="paddle.matmul",
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    y_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    y_static,
                    self.transpose_x,
                    self.transpose_y,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "y": self.np_y, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # save static res for test_matmul_incubate
        np.savez(
            self.save_static_res_path,
            out_static=out_static,
            out_grads_static_0=out_grads_static[0],
            out_grads_static_1=out_grads_static[1],
        )
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
            api="paddle.matmul",
        )
        # compare develop static backward res with torch
        for idx in range(len(out_grads_static)):
            np_assert_accuracy(
                out_grads_static[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="backward",
                api="paddle.matmul",
            )

    def test_eager_stability(self):
        x_eager, y_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, y_eager, self.transpose_x, self.transpose_y, dout_eager
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
                x_eager, y_eager, self.transpose_x, self.transpose_y, dout_eager
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
                api="paddle.matmul",
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
                    api="paddle.matmul",
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    y_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    y_static,
                    self.transpose_x,
                    self.transpose_y,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "y": self.np_y, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(5):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "y": self.np_y, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                # test develop static forward stability
                np_assert_staility(
                    out_static,
                    out_static_baseline,
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="paddle.matmul",
                )
                # test develop static backward stability
                for idx in range(len(out_grads_static)):
                    np_assert_staility(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="backward",
                        api="paddle.matmul",
                    )


class TestMatmulDevelopCase1_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"


class TestMatmulDevelopCase1_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"


class TestMatmulDevelopCase2_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp32.npz"


class TestMatmulDevelopCase2_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp16.npz"


class TestMatmulDevelopCase2_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npz"

class TestMatmulDevelopCase3_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case3_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_fp32.npz"


class TestMatmulDevelopCase3_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case3_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_fp16.npz"


class TestMatmulDevelopCase3_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case3_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_bfp16.npz"

class TestMatmulDevelopCase4_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case4.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case4_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case4_fp32.npz"


class TestMatmulDevelopCase4_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case4.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case4_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case4_fp16.npz"


class TestMatmulDevelopCase4_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case4.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case4_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case4_bfp16.npz"

class TestMatmulDevelopCase5_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case5.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case5_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case5_fp32.npz"


class TestMatmulDevelopCase5_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case5.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case5_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case5_fp16.npz"


class TestMatmulDevelopCase5_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case5.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case5_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case5_bfp16.npz"

class TestMatmulDevelopCase6_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case6.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case6_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case6_fp32.npz"


class TestMatmulDevelopCase6_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case6.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case6_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case6_fp16.npz"


class TestMatmulDevelopCase6_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case6.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case6_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case6_bfp16.npz"

class TestMatmulDevelopCase7_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case7.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case7_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case7_fp32.npz"


class TestMatmulDevelopCase7_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case7.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case7_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case7_fp16.npz"


class TestMatmulDevelopCase7_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case7.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case7_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case7_bfp16.npz"


class TestMatmulDevelopCase8_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case8.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case8_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case8_fp32.npz"


class TestMatmulDevelopCase8_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case8.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case8_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case8_fp16.npz"


class TestMatmulDevelopCase8_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case8.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case8_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case8_bfp16.npz"

class TestMatmulDevelopCase9_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case9.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case9_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case9_fp32.npz"


class TestMatmulDevelopCase9_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case9.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case9_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case9_fp16.npz"

class TestMatmulDevelopCase9_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case9.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case9_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case9_bfp16.npz"

class TestMatmulDevelopCase10_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case10.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case10_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case10_fp32.npz"


class TestMatmulDevelopCase10_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case10.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case10_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case10_fp16.npz"

class TestMatmulDevelopCase10_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case10.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case10_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case10_bfp16.npz"

class TestMatmulDevelopCase11_FP32(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case11.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case11_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case11_fp32.npz"


class TestMatmulDevelopCase11_FP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case11.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case11_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case11_fp16.npz"

class TestMatmulDevelopCase11_BFP16(TestMatmulDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case11.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case11_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case11_bfp16.npz"


class TestMatmulAndAddDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, y_torch, b_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, y_torch, self.transpose_x, self.transpose_y, b_torch, dout_torch
        )
        del x_torch
        del y_torch
        del dout_torch
        del b_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.np_input_dir = "./inputs_case12.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case12_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case12_fp32.npz"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_x = np_inputs_array["x"]
        self.np_y = np_inputs_array["y"]
        self.np_b = np_inputs_array["b"]
        self.np_dout = np_inputs_array["dout"]
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_y = self.np_y.astype("float16")
            self.np_dout = self.np_dout.astype("float16")
            self.np_b = self.np_b.astype("float16")

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
        b_torch = torch.tensor(
            self.np_b,
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
        return x_torch, y_torch, b_torch, dout_torch

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
        b_eager = paddle.to_tensor(
            self.np_b,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        b_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, y_eager, b_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        y_static = paddle.static.data(
            'y',
            shape=self.np_y.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        y_static.stop_gradient = False
        b_static = paddle.static.data(
            'b',
            shape=self.np_b.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        b_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, y_static, b_static, dout_static

    def cal_torch_res(self, x, y, transpose_x, transpose_y, b,dout):
        x_t = x
        y_t = y
        b_t = b
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            y_t = y.to(dtype=torch.bfloat16)
            b_t = b.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        if transpose_x:
            x_t = torch.transpose(x_t, -1, -2)
        if transpose_y:
            y_t = torch.transpose(y_t, -1, -2)
        out = torch.matmul(x_t, y_t)
        z = torch.add(out, b_t)
        out_grads = torch.autograd.grad([z], [x, y], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, x, y, transpose_x, transpose_y,b, dout):
        x_t = x
        y_t = y
        b_t = b
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            y_t = paddle.cast(y, dtype="uint16")
            b_t = paddle.cast(b, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.matmul(x_t, y_t, transpose_x, transpose_y)
        z = paddle.add(out, b_t)
        out_grads = paddle.grad([z], [x, y], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def cal_static_res(self, x, y, transpose_x, transpose_y, b, dout):
        x_t = x
        y_t = y
        b_t = b
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            y_t = paddle.cast(y, dtype="uint16")
            b_t = paddle.cast(b, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.matmul(x_t, y_t, transpose_x, transpose_y)
        z = paddle.add(out, b_t)
        out_grads = paddle.static.gradients(
            [z], [x, y], target_gradients=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, y_eager, b_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, y_eager, self.transpose_x, self.transpose_y, b_eager, dout_eager
        )
        del x_eager
        del b_eager
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
        # save eager res for test_matmul_incubate
        np.savez(
            self.save_eager_res_path,
            out_eager=out_eager_np,
            out_grads_eager_0=out_grads_eager_np[0],
            out_grads_eager_1=out_grads_eager_np[1],
        )
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
            api="paddle.matmul",
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
                api="paddle.matmul",
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    y_static,
                    b_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    y_static,
                    self.transpose_x,
                    self.transpose_y,
                    b_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "y": self.np_y,"b":self.np_b, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # save static res for test_matmul_incubate
        np.savez(
            self.save_static_res_path,
            out_static=out_static,
            out_grads_static_0=out_grads_static[0],
            out_grads_static_1=out_grads_static[1],
        )
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
            api="paddle.matmul",
        )
        # compare develop static backward res with torch
        for idx in range(len(out_grads_static)):
            np_assert_accuracy(
                out_grads_static[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="backward",
                api="paddle.matmul",
            )

    def test_eager_stability(self):
        x_eager, y_eager, b_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, y_eager, self.transpose_x, self.transpose_y, b_eager, dout_eager
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
                x_eager, y_eager, self.transpose_x, self.transpose_y, b_eager, dout_eager
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
                api="paddle.matmul",
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
                    api="paddle.matmul",
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    y_static,
                    b_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    y_static,
                    self.transpose_x,
                    self.transpose_y,
                    b_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "y": self.np_y, "b": self.np_b, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(5):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "y": self.np_y,  "b": self.np_b, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                # test develop static forward stability
                np_assert_staility(
                    out_static,
                    out_static_baseline,
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="paddle.matmul",
                )
                # test develop static backward stability
                for idx in range(len(out_grads_static)):
                    np_assert_staility(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="backward",
                        api="paddle.matmul",
                    )

class TestMatmulAndAddDevelopCase1_FP16(TestMatmulAndAddDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case12.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case12_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case12_fp16.npz"

class TestMatmulAndAddDevelopCase1_BFP16(TestMatmulAndAddDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case12.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case12_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case12_bfp16.npz"


class TestMatmulFP32vsBFP16_case10(unittest.TestCase):
    def test_fp32vsbfp16(self):
        res_fp32 = np.load('eager_develop_res_case10_fp32.npz')
        out_eager_fp32 = res_fp32["out_eager"]
        out_eager_grad_0_fp32 = res_fp32["out_grads_eager_0"]
        out_eager_grad_1_fp32 = res_fp32["out_grads_eager_1"]
        res_bfp16 = np.load('eager_develop_res_case10_bfp16.npz')
        out_eager_bfp16 = res_bfp16["out_eager"]
        out_eager_grad_0_bfp16 = res_bfp16["out_grads_eager_0"]
        out_eager_grad_1_bfp16 = res_bfp16["out_grads_eager_1"]
        try:
            np.testing.assert_equal(out_eager_fp32, out_eager_bfp16)
        except Exception as e:
            print(e)
        try:
            np.testing.assert_equal(out_eager_grad_0_fp32, out_eager_grad_0_bfp16)
        except Exception as e:
            print(e)
        try:
            np.testing.assert_equal(out_eager_grad_1_fp32, out_eager_grad_1_bfp16)
        except Exception as e:
            print(e)
if __name__ == '__main__':
    generate_np_inputs_and_dout()
    unittest.main()
