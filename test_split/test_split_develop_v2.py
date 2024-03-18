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

class TestSplitDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, dim_torch, num_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, dim_torch, num_torch, dout_torch
        )
        del x_torch
        del dim_torch
        del num_torch
        del dout_torch
        self.out_torch = map_structure(
            lambda x: x.cpu().detach().numpy(),
            out_torch,
        )
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
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
        self.np_x = np.random.random(size=[1, 1, 4096, 128]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 2
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[1, 1, 4096, 64]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dim_torch = self.np_dim
        num_torch = int((self.np_x.shape[self.np_dim])/self.np_num)
        dout_torch = []
        for dout in self.np_dout:
            dout_torch.append(
                torch.tensor(
                dout,
                device='cuda',
                dtype=convert_dtype_to_torch_type(self.dtype)
                if self.dtype != 'bfloat16'
                else torch.float32,
                requires_grad=True,
                )
            )
        return x_torch, dim_torch, num_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dim_eager = self.np_dim
        num_eager = self.np_num
        dout_eager = []
        for dout in self.np_dout:
            dout_eager.append(
                paddle.to_tensor(
                dout,
                dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
                place="gpu",
                )
            )
        for dout in dout_eager:
            dout.stop_gradient = False
        return x_eager, dim_eager, num_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        dim_static = self.np_dim
        num_static = self.np_num
        dout_static = []
        dout_name = ('dout0', 'dout1', 'dout2')
        for i in range(self.np_num):
            dout_static.append(
                paddle.static.data(
                dout_name[i],
                shape=self.np_dout[i].shape,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
                )
            )
        for dout in dout_static:
            dout.stop_gradient = False
        return x_static, dim_static, num_static, dout_static

    def cal_torch_res(self, x, dim, num, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            for dout_t in dout:
                dout_t = dout_t.to(dtype=torch.bfloat16)
        out = torch.split(x, num, dim)
        out_grads = torch.autograd.grad(out, [x], grad_outputs=dout)
        if self.dtype == "bfloat16":
            out = map_structure(lambda x: x.to(dtype=torch.float32), out)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, dim, num, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            for dout_t in dout:
                dout_t = paddle.cast(dout_t, dtype="uint16")
        out = paddle.split(x, num, dim)
        out_grads = paddle.grad(
            out, [x], grad_outputs=dout
        )
        if self.dtype == "bfloat16":
            out = map_structure(lambda x: paddle.cast(x, dtype='float32'), out)
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def cal_static_res(self, x, dim, num, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            for dout_t in dout:
                dout_t = paddle.cast(dout_t, dtype="uint16")
        out = paddle.split(x, num, dim)
        out_grads = paddle.static.gradients(
            out, [x], target_gradients=dout
        )
        if self.dtype == "bfloat16":
            out = map_structure(lambda x: paddle.cast(x, dtype='float32'), out)
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, dim_eager, num_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, dim_eager, num_eager, dout_eager
        )
        del x_eager
        del dim_eager
        del num_eager
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = map_structure(
            lambda x: x.numpy(),
            out_eager,
        )
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()
        # compare develop eager forward res with torch
        for idx in range(len(out_eager_np)):
            np_assert_accuracy(
                out_eager_np[idx],
                self.out_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.split",
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
                api="paddle.split",
            )

    def test_static_accuracy(self):
        with paddle.base.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dim_static,
                    num_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    dim_static,
                    num_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            if self.np_num == 2:
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout0": self.np_dout[0], "dout1": self.np_dout[1]},
                    fetch_list=out_static + out_grads_static,
                )
                out_static, out_grads_static = out[:2], out[2:]
            else:
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout0": self.np_dout[0], "dout1": self.np_dout[1], "dout2": self.np_dout[2]},
                    fetch_list=out_static + out_grads_static,
                )
                out_static, out_grads_static = out[:3], out[3:]

        # compare develop static forward res with torch
        for idx in range(len(out_static)):
            np_assert_accuracy(
                out_static[idx],
                self.out_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="forward",
                api="paddle.split",
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
                api="paddle.split",
            )

    def test_eager_stability(self):
        x_eager, dim_eager, num_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, dim_eager, num_eager, dout_eager
        )
        out_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_eager_baseline,
        )
        out_grads_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager_baseline,
        )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager, out_grads_eager = self.cal_eager_res(
                x_eager, dim_eager, num_eager, dout_eager
            )
            out_eager = map_structure(
                lambda x: x.numpy(),
                out_eager,
            )
            out_grads_eager = map_structure(
                lambda x: x.numpy(),
                out_grads_eager,
            )
            # test develop eager forward stability
            for idx in range(len(out_eager)):
                np_assert_staility(
                    out_eager[idx],
                    out_eager_baseline_np[idx],
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="paddle.split",
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
                    api="paddle.split",
                )

    def test_static_stability(self):
        with paddle.base.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dim_static,
                    num_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    dim_static,
                    num_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            if self.np_num == 2:
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout0": self.np_dout[0], "dout1": self.np_dout[1]},
                    fetch_list=out_static_pg + out_grads_static_pg,
                )
                out_static_baseline, out_grads_static_baseline = out[:2], out[2:]
            else :
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout0": self.np_dout[0], "dout1": self.np_dout[1], "dout2": self.np_dout[2]},
                    fetch_list=out_static_pg + out_grads_static_pg,
                )
                out_static_baseline, out_grads_static_baseline = out[:3], out[3:]
            for i in range(5):
                if self.np_num == 2:
                    out = exe.run(
                        mp,
                        feed={"x": self.np_x, "dout0": self.np_dout[0], "dout1": self.np_dout[1]},
                        fetch_list=out_static_pg + out_grads_static_pg,
                    )
                    out_static, out_grads_static = out[:2], out[2:]
                else :
                    out = exe.run(
                        mp,
                        feed={"x": self.np_x, "dout0": self.np_dout[0], "dout1": self.np_dout[1], "dout2": self.np_dout[2]},
                        fetch_list=out_static_pg + out_grads_static_pg,
                    )
                    out_static, out_grads_static = out[:3], out[3:]
                # test develop static forward stability
                for idx in range(len(out_static)):
                    np_assert_staility(
                        out_static[idx],
                        out_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="forward",
                        api="paddle.split",
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
                        api="paddle.split",
                    )


class TestSplitDevelopCase1_FP16(TestSplitDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase1_BFP16(TestSplitDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSplitDevelopCase2_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 1, 10944]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 2
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[4096, 1, 5472]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase2_FP16(TestSplitDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase2_BFP16(TestSplitDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestSplitDevelopCase3_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 1, 16, 384]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 3
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[4096, 1, 16, 128]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase3_FP16(TestSplitDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase3_BFP16(TestSplitDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestSplitDevelopCase4_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 1, 128]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 2
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[1, 8192, 1, 64]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase4_FP16(TestSplitDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase4_BFP16(TestSplitDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestSplitDevelopCase5_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 14, 384]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 3
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[1, 8192, 14, 128]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase5_FP16(TestSplitDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase5_BFP16(TestSplitDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestSplitDevelopCase6_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 9632]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 2
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[1, 8192, 4816]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase6_FP16(TestSplitDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase6_BFP16(TestSplitDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestSplitDevelopCase7_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 10, 32, 384]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 3
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[256, 10, 32, 128]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase7_FP16(TestSplitDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase7_BFP16(TestSplitDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestSplitDevelopCase8_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 10, 32, 64]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 2
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[256, 10, 32, 32]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase8_FP16(TestSplitDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase8_BFP16(TestSplitDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSplitDevelopCase9_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8, 40, 72]).astype("float32") - 0.5
        self.np_dim = 1
        self.np_num = 2
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[1, 4, 40, 72]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase9_FP16(TestSplitDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase9_BFP16(TestSplitDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSplitDevelopCase10_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 181, 4608]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 3
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[10, 181, 1536]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase10_FP16(TestSplitDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase10_BFP16(TestSplitDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestSplitDevelopCase11_FP32(TestSplitDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 10, 32, 128]).astype("float32") - 0.5
        self.np_dim = -1
        self.np_num = 2
        self.np_dout = []
        for _ in range(self.np_num):
            self.np_dout.append(np.random.random(size=[256, 10, 32, 64]).astype("float32") - 0.5)
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            for i in range(self.np_num):
                self.np_dout[i] = self.np_dout[i].astype("float16")

class TestSplitDevelopCase11_FP16(TestSplitDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestSplitDevelopCase11_BFP16(TestSplitDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
