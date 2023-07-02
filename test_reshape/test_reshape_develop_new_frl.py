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

class TestReshapeDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.shape_tensor = False
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
        self.np_x = np.random.random(size=[1, 1, 4096, 64, 2]).astype("float32") - 0.5
        self.np_shape = [1, 1, 4096, 128]
        self.np_dout = np.random.random(size=[1, 1, 4096, 128]).astype("float32") - 0.5
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
        if self.shape_tensor:
            shape_torch = self.np_shape.tolist()
        else:
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
        if self.shape_tensor:
            shape_eager = paddle.to_tensor(
                self.np_shape,
                dtype="int32",
                place="gpu",
            )
        else:
            shape_eager = self.np_shape
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, shape_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        if self.shape_tensor:
            shape_static = paddle.static.data(
                'shape',
                shape=self.np_shape.shape,
                dtype="int32",
            )
            dout_static = paddle.static.data(
                'dout',
                shape=(-1, -1, -1, -1),
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            )
        else:
            shape_static = self.np_shape
            dout_static = paddle.static.data(
                'dout',
                shape=self.np_dout.shape,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            )
        dout_static.stop_gradient = False
        return x_static, shape_static, dout_static

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
        out = paddle.reshape(x, shape)
        out_grads = paddle.grad(
            [out], [x], grad_outputs=[dout]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def cal_static_res(self, x, shape, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.reshape(x, shape)
        out_grads = paddle.static.gradients(
            [out], [x], target_gradients=[dout]
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
            api="paddle.reshape",
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
                api="paddle.reshape",
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    shape_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    shape_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            if self.shape_tensor:
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "shape": self.np_shape, "dout": self.np_dout},
                    fetch_list=[out_static] + out_grads_static,
                )
            else:
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
            api="paddle.reshape",
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
                api="paddle.reshape",
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
                api="paddle.reshape",
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
                    api="paddle.reshape",
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    shape_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    shape_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            if self.shape_tensor:
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "shape": self.np_shape, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
            else:
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(5):
                if self.shape_tensor:
                    out = exe.run(
                        mp,
                        feed={"x": self.np_x, "shape": self.np_shape, "dout": self.np_dout},
                        fetch_list=[out_static_pg] + out_grads_static_pg,
                    )
                else:
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
                    api="paddle.reshape",
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
                        api="paddle.reshape",
                    )


class TestReshapeDevelopCase1_FP16(TestReshapeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestReshapeDevelopCase1_BFP16(TestReshapeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase2_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 16, 4096, 64, 2]).astype("float32") - 0.5
        self.np_shape = np.array([1, 16, 4096, 128]).astype("int32")
        self.shape_tensor = True
        self.np_dout = np.random.random(size=[1, 16, 4096, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase2_FP16(TestReshapeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase2_BFP16(TestReshapeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase3_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 4096, 16, 128]).astype("float32") - 0.5
        self.np_shape = [0, 0, 2048]
        self.np_dout = np.random.random(size=[1, 4096, 2048]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase3_FP16(TestReshapeDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase3_BFP16(TestReshapeDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase4_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 128]).astype("float32") - 0.5
        self.np_shape = (1, 1, 4096, 128)
        self.np_dout = np.random.random(size=[1, 1, 4096, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase4_FP16(TestReshapeDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase4_BFP16(TestReshapeDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase5_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096]).astype("float32") - 0.5
        self.np_shape = [4096]
        self.np_dout = np.random.random(size=[4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase5_FP16(TestReshapeDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase5_BFP16(TestReshapeDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase6_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[8192]).astype("float32") - 0.5
        self.np_shape = [8192]
        self.np_dout = np.random.random(size=[8192]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase6_FP16(TestReshapeDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase6_BFP16(TestReshapeDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase7_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 1, 64, 2]).astype("float32") - 0.5
        self.np_shape = [1, 8192, 1, 128]
        self.np_dout = np.random.random(size=[1, 8192, 1, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase7_FP16(TestReshapeDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase7_BFP16(TestReshapeDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase8_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 14, 128]).astype("float32") - 0.5
        self.np_shape = [1, 8192, 1792]
        self.np_dout = np.random.random(size=[1, 8192, 1792]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase8_FP16(TestReshapeDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase8_BFP16(TestReshapeDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase9_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 1]).astype("float32") - 0.5
        self.np_shape = [8192]
        self.np_dout = np.random.random(size=[8192]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase9_FP16(TestReshapeDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase9_BFP16(TestReshapeDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase10_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192, 5376]).astype("float32") - 0.5
        self.np_shape = [1, 8192, 14, 384]
        self.np_dout = np.random.random(size=[1, 8192, 14, 384]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase9_FP16(TestReshapeDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase9_BFP16(TestReshapeDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase11_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 8192]).astype("float32") - 0.5
        self.np_shape = [8192]
        self.np_dout = np.random.random(size=[8192]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase11_FP16(TestReshapeDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase11_BFP16(TestReshapeDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase12_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 31250]).astype("float32") - 0.5
        self.np_shape = [448000000]
        self.np_dout = np.random.random(size=[448000000]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase12_FP16(TestReshapeDevelopCase12_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase12_BFP16(TestReshapeDevelopCase12_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase13_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 5376]).astype("float32") - 0.5
        self.np_shape = [77070336]
        self.np_dout = np.random.random(size=[77070336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase13_FP16(TestReshapeDevelopCase13_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase13_BFP16(TestReshapeDevelopCase13_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase14_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336, 9632]).astype("float32") - 0.5
        self.np_shape = [138084352]
        self.np_dout = np.random.random(size=[138084352]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase14_FP16(TestReshapeDevelopCase14_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase14_BFP16(TestReshapeDevelopCase14_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestReshapeDevelopCase15_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[14336]).astype("float32") - 0.5
        self.np_shape = [14336]
        self.np_dout = np.random.random(size=[14336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase15_FP16(TestReshapeDevelopCase15_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase15_BFP16(TestReshapeDevelopCase15_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase16_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1792, 14336]).astype("float32") - 0.5
        self.np_shape = [25690112]
        self.np_dout = np.random.random(size=[25690112]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase16_FP16(TestReshapeDevelopCase16_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase16_BFP16(TestReshapeDevelopCase16_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase17_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[31250, 14336]).astype("float32") - 0.5
        self.np_shape = [448000000]
        self.np_dout = np.random.random(size=[448000000]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase17_FP16(TestReshapeDevelopCase17_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase17_BFP16(TestReshapeDevelopCase17_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase18_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[31250]).astype("float32") - 0.5
        self.np_shape = [31250]
        self.np_dout = np.random.random(size=[31250]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase18_FP16(TestReshapeDevelopCase18_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase18_BFP16(TestReshapeDevelopCase18_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase19_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4816, 14336]).astype("float32") - 0.5
        self.np_shape = [69042176]
        self.np_dout = np.random.random(size=[69042176]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase19_FP16(TestReshapeDevelopCase19_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase19_BFP16(TestReshapeDevelopCase19_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase20_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[5376]).astype("float32") - 0.5
        self.np_shape = [5376]
        self.np_dout = np.random.random(size=[5376]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase20_FP16(TestReshapeDevelopCase20_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase20_BFP16(TestReshapeDevelopCase20_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase21_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[8192, 128]).astype("float32") - 0.5
        self.np_shape = [1, 1, 8192, 128]
        self.np_dout = np.random.random(size=[1, 1, 8192, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase21_FP16(TestReshapeDevelopCase21_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase21_BFP16(TestReshapeDevelopCase21_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase22_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[9632]).astype("float32") - 0.5
        self.np_shape = [9632]
        self.np_dout = np.random.random(size=[9632]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase22_FP16(TestReshapeDevelopCase22_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase22_BFP16(TestReshapeDevelopCase22_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
