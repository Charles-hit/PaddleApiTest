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
from tqdm import tqdm

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

class TestSliceDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, dout_torch
        )
        del x_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.start = 0
        self.end = 1

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[64, 4096]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1, 4096]).astype("float32") - 0.5
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
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, dout_static

    def cal_torch_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        out = x[self.start : self.end]
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = x[self.start : self.end]
        out_grads = paddle.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def cal_static_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = x[self.start : self.end]
        out_grads = paddle.static.gradients(
            [out], [x], target_gradients=[dout]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, dout_eager
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
            api="paddle.slice",
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
                api="paddle.slice",
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

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
            api="paddle.slice",
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
                api="paddle.slice",
            )

    def test_eager_stability(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, dout_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager_baseline,
        )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()
        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(
                x_eager, dout_eager
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
                api="paddle.slice",
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
                    api="paddle.slice",
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout": self.np_dout},
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
                    api="paddle.slice",
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
                        api="paddle.slice",
                    )

class TestSliceDevelopCase1_FP16(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start = 0
        self.end = 1

class TestSliceDevelopCase1_BFP16(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start = 0
        self.end = 1

class TestSliceDevelopCase2_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start = 63
        self.end = 64

class TestSliceDevelopCase2_FP16(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start = 63
        self.end = 64

class TestSliceDevelopCase2_BFP16(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start = 63
        self.end = 64

class TestSliceDevelopCase3_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start = 25
        self.end = 26

class TestSliceDevelopCase3_FP16(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start = 25
        self.end = 26

class TestSliceDevelopCase3_BFP16(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start = 25
        self.end = 26

class TestSliceDevelopCase4_FP32(TestSliceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[64, 4096, 4096]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1, 4096, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")
            
    def init_params(self):
        self.dtype = "float32"
        self.start = 0
        self.end = 1

class TestSliceDevelopCase4_FP16(TestSliceDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start = 0
        self.end = 1

class TestSliceDevelopCase4_BFP16(TestSliceDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start = 0
        self.end = 1

class TestSliceDevelopCase5_FP32(TestSliceDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start = 25
        self.end = 26

class TestSliceDevelopCase5_FP16(TestSliceDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start = 25
        self.end = 26

class TestSliceDevelopCase5_BFP16(TestSliceDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start = 25
        self.end = 26

class TestSliceDevelopCase6_FP32(TestSliceDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start = 63
        self.end = 64

class TestSliceDevelopCase6_FP16(TestSliceDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start = 63
        self.end = 64

class TestSliceDevelopCase6_BFP16(TestSliceDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start = 63
        self.end = 64

class TestSliceDevelopCase7_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.axes = [0]
        self.starts = [0]
        self.ends = [2048]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 1, 4096]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[2048, 1, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def cal_torch_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        out = x[0:2048]
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, dout, axes, starts, ends):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.slice(x, axes, starts, ends)
        out_grads = paddle.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def cal_static_res(self, x, dout, axes, starts, ends):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.slice(x, axes, starts, ends)
        out_grads = paddle.static.gradients(
            [out], [x], target_gradients=[dout]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, dout_eager, self.axes, self.starts, self.ends
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
            api="paddle.slice",
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
                api="paddle.slice",
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    dout_static,
                    self.axes, self.starts, self.ends
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

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
            api="paddle.slice",
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
                api="paddle.slice",
            )

    def test_eager_stability(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, dout_eager,self.axes, self.starts, self.ends
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager_baseline,
        )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()
        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(
                x_eager, dout_eager, self.axes, self.starts, self.ends
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
                api="paddle.slice",
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
                    api="paddle.slice",
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    dout_static,
                    self.axes, self.starts, self.ends
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout": self.np_dout},
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
                    api="paddle.slice",
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
                        api="paddle.slice",
                    )


class TestSliceDevelopCase7_FP16(TestSliceDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.axes = [0]
        self.starts = [0]
        self.ends = [2048]


class TestSliceDevelopCase7_BFP16(TestSliceDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.axes = [0]
        self.starts = [0]
        self.ends = [2048]

class TestSliceIntItemCase8(unittest.TestCase):
    def setUp(self) -> None:
        self.case_path = '/workspace/PaddleApiTest/test_slice/slice_286239_mp1_comm.log'

    def init_item(self, new_line):
        self.np_item = new_line[1]
        self.torch_item = new_line[1]
        self.eager_item = new_line[1]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs(self):
        # init np array 
        self.np_x = np.random.random(size=self.shape).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def init_dout_shape(self):
        try:
            out = self.np_x[self.np_item]
        except:
            breakpoint()
        self.dout_shape = out.shape

    def init_np_dout(self):
        self.np_dout = np.random.random(size=self.dout_shape).astype("float32") - 0.5
        if self.dtype == "float16":
            self.np_dout = self.np_dout.astype("float16")

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, dout_eager

    def cal_eager_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = x[self.eager_item]
        if out.shape != dout.shape:
            dout = paddle.reshape(dout, out.shape)
        try:
            out_grads = paddle.grad([out], [x], grad_outputs=[dout])
        except:
            breakpoint()
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
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
        return x_torch, dout_torch

    def cal_torch_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        out = x[self.torch_item]
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        np.random.seed(2023)
        with open(self.case_path, 'r') as f:
            for line in tqdm(f.readlines()):
                for dtype in ["float16", "float32", "bfloat16"]:
                    # some cases are error so skip them
                    if '#' in line:
                        continue
                    self.dtype = dtype
                    # (shape, item)
                    new_line = line.strip().strip(',')
                    new_line = eval(new_line)
                    self.shape = new_line[0]
                    if not isinstance(self.shape, tuple):
                        self.shape = (self.shape, ) 

                    self.init_item(new_line)                       

                    self.init_threshold()
                    self.init_np_inputs()
                    self.init_dout_shape()
                    self.init_np_dout()

                    x_torch, dout_torch = self.gen_torch_inputs_and_dout()
                    out_torch, out_grads_torch = self.cal_torch_res(
                        x_torch, dout_torch
                    )
                    del x_torch
                    del dout_torch
                    self.out_torch = out_torch.cpu().detach().numpy()
                    self.out_grads_torch = map_structure(
                        lambda x: x.cpu().numpy(),
                        out_grads_torch,
                    )
                    del out_torch, out_grads_torch
                    torch.cuda.empty_cache()

                    x_eager, dout_eager = self.gen_eager_inputs_and_dout()
                    out_eager, out_grads_eager = self.cal_eager_res(
                        x_eager, dout_eager
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
                        out_eager_np.flatten(),
                        self.out_torch.flatten(),
                        self.atol,
                        self.rtol,
                        self.dtype,
                        version_a="paddle_develop",
                        version_b="torch",
                        eager_or_static_mode="eager",
                        fwd_or_bkd="forward",
                        api="paddle.slice",
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
                            api="paddle.slice",
                        )

class TestSlicePySliceItemCase8(TestSliceIntItemCase8):
    def setUp(self) -> None:
        self.case_path = '/workspace/PaddleApiTest/test_slice/slice_286239_mp1_slice_item.log'

class TestSliceTensorItemCase8(TestSliceIntItemCase8):
    def setUp(self) -> None:
        self.case_path = '/workspace/PaddleApiTest/test_slice/slice_286239_mp1_tensor_item.log'
    
    def init_item(self, new_line):
        all_np_item = []
        all_torch_item = []
        all_eager_item = []
        for (shape, value) in new_line[1]:
            np_item = np.array(value).reshape(shape)
            torch_item = torch.tensor(np_item, device='cuda')
            eager_item = paddle.to_tensor(np_item, place="gpu")
            all_np_item.append(np_item)
            all_torch_item.append(torch_item)
            all_eager_item.append(eager_item)
        if len(all_np_item) == 1:
            self.np_item = all_np_item[0]
            self.torch_item = all_torch_item[0]
            self.eager_item = all_eager_item[0]
        elif len(all_np_item) > 1:
            self.np_item = tuple(all_np_item)
            self.torch_item = tuple(all_torch_item)
            self.eager_item = tuple(all_eager_item)

class TestSliceIntItemCase9(TestSliceIntItemCase8):
    def setUp(self) -> None:
        self.case_path = '/workspace/PaddleApiTest/test_slice/slice_415318_mp8_comm.log'

class TestSlicePySliceItemCase9(TestSlicePySliceItemCase8):
    def setUp(self) -> None:
        self.case_path = '/workspace/PaddleApiTest/test_slice/slice_415318_mp8_slice_item.log'

class TestSliceTensorItemCase9(TestSliceTensorItemCase8):
    def setUp(self) -> None:
        self.case_path = '/workspace/PaddleApiTest/test_slice/slice_415318_mp8_tensor_item.log'

if __name__ == '__main__':
    unittest.main()