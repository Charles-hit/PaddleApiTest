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
        with paddle.base.framework._dygraph_guard(None):
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
        with paddle.base.framework._dygraph_guard(None):
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

# ---

# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [128, 256, 3, 3] }, }, params: [ shape: [294912], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1000] }, }, params: [ shape: [1000], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [512] }, }, params: [ shape: [512], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [12288] }, }, params: [ shape: [12288], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1, 512, 2880] }, }, params: [ shape: [1, 512, 40, 72], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.base.framework.EagerParamBase'>, shape: [8] }, }, params: [ shape: [1, -1, 1, 1], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 181, 16, 96] }, }, params: [ shape: [0, 0, 1536], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [4] }, }, params: [ shape: [4], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [416, 576, 3] }, }, params: [ shape: [416, 576, 3], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [576, 352, 3] }, }, params: [ shape: [576, 352, 3], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 1, 16, 96] }, }, params: [ shape: [10, 16, 96], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256, 512, 3, 3] }, }, params: [ shape: [1179648], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 180, 16, 96] }, }, params: [ shape: [10, 1, 10, 18, 16, 96], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [8, 8, 1, 1] }, }, params: [ shape: [64], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [4, 4, 1, 1] }, }, params: [ shape: [16], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1536, 1536] }, }, params: [ shape: [2359296], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [3072, 1536] }, }, params: [ shape: [4718592], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 1, 16, 96] }, }, params: [ shape: [10, 1, 16, 96], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [4096] }, }, params: [ shape: [4096], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 256, 1536] }, }, params: [ shape: [0, 0, 16, 96], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.base.framework.EagerParamBase'>, shape: [4] }, }, params: [ shape: [1, -1, 1, 1, 1], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1, 2880, 512] }, }, params: [ shape: [0, 0, 1, 512], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256, 10, 32, 128] }, }, params: [ shape: [256, 10, 4096], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [4096, 16384] }, }, params: [ shape: [67108864], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 4, 40, 72] }, }, params: [ shape: [10, 1, 4, 40, 72], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [320, 256, 256] }, }, params: [ shape: [10, 32, 256, 256], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [576, 576, 3] }, }, params: [ shape: [576, 576, 3], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 180, 64] }, }, params: [ shape: [10, 1, 10, 18, 1, 4, 4, 4], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [320, 256, 128] }, }, params: [ shape: [10, 32, 256, 128], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1536, 4096] }, }, params: [ shape: [6291456], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256, 256, 3, 3] }, }, params: [ shape: [589824], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1536, 64] }, }, params: [ shape: [98304], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [130528, 4096] }, }, params: [ shape: [534642688], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [64] }, }, params: [ shape: [64], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [128, 256, 1, 1] }, }, params: [ shape: [32768], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 32, 256, 256] }, }, params: [ shape: [320, 256, -1], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [4096, 4096] }, }, params: [ shape: [16777216], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [16] }, }, params: [ shape: [16], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [320, 576, 3] }, }, params: [ shape: [320, 576, 3], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 4, 1, 1, 10, 4, 18, 4] }, }, params: [ shape: [10, 4, 1, 40, 72], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256, 512, 1, 1] }, }, params: [ shape: [131072], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [576, 416, 3] }, }, params: [ shape: [576, 416, 3], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [512, 256, 1, 1] }, }, params: [ shape: [131072], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256, 10, 12288] }, }, params: [ shape: [256, 10, 32, 384], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [3] }, }, params: [ shape: [3], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [4, 4, 3, 3, 3] }, }, params: [ shape: [432], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [3, 128, 3, 3] }, }, params: [ shape: [3456], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [8] }, }, params: [ shape: [8], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [128] }, }, params: [ shape: [128], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256, 128, 1, 1] }, }, params: [ shape: [32768], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [3] }, }, params: [ shape: [-1, 1, 1], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [352, 576, 3] }, }, params: [ shape: [352, 576, 3], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [16384] }, }, params: [ shape: [16384], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [4096, 1536] }, }, params: [ shape: [6291456], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1536] }, }, params: [ shape: [1536], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256, 10, 32, 128] }, }, params: [ shape: [256, 320, -1], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1536, 4, 1, 4, 4] }, }, params: [ shape: [98304], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [512, 512] }, }, params: [ shape: [262144], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 181, 1536] }, }, params: [ shape: [0, 0, 16, 96], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 1, 3, 320, 576] }, }, params: [ shape: [10, 3, 320, 576], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [10, 1, 10, 18, 16, 96] }, }, params: [ shape: [10, 180, 16, 96], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [128, 128, 3, 3] }, }, params: [ shape: [147456], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [512, 256, 3, 3] }, }, params: [ shape: [1179648], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1536, 4608] }, }, params: [ shape: [7077888], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [8, 512, 3, 3] }, }, params: [ shape: [36864], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256, 128, 3, 3] }, }, params: [ shape: [294912], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [512, 4, 3, 3] }, }, params: [ shape: [18432], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [4096, 12288] }, }, params: [ shape: [50331648], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [16384, 4096] }, }, params: [ shape: [67108864], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [512, 512, 3, 3] }, }, params: [ shape: [2359296], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.base.framework.EagerParamBase'>, shape: [1536] }, }, params: [ shape: [1, -1, 1, 1, 1], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1, 512, 40, 72] }, }, params: [ shape: [1, 512, 2880], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.base.framework.EagerParamBase'>, shape: [128] }, }, params: [ shape: [1, -1, 1, 1], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [128, 3, 3, 3] }, }, params: [ shape: [3456], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [1, 2880, 1, 512] }, }, params: [ shape: [0, 0, 512], ]}
# {function_name : reshape, inputs: { { x, type: <class 'paddle.Tensor'>, shape: [256] }, }, params: [ shape: [256], ]}


class TestReshapeDevelopCase23_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[128, 256, 3, 3]).astype("float32") - 0.5
        self.np_shape = [294912]
        self.np_dout = np.random.random(size=[294912]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase23_FP16(TestReshapeDevelopCase23_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase23_BFP16(TestReshapeDevelopCase23_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase24_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1000]).astype("float32") - 0.5
        self.np_shape = [1000]
        self.np_dout = np.random.random(size=[1000]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase24_FP16(TestReshapeDevelopCase24_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase24_BFP16(TestReshapeDevelopCase24_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase25_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[512]).astype("float32") - 0.5
        self.np_shape = [512]
        self.np_dout = np.random.random(size=[512]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase25_FP16(TestReshapeDevelopCase25_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase25_BFP16(TestReshapeDevelopCase25_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase26_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[12288]).astype("float32") - 0.5
        self.np_shape = [12288]
        self.np_dout = np.random.random(size=[12288]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase26_FP16(TestReshapeDevelopCase26_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase26_BFP16(TestReshapeDevelopCase26_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase27_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 512, 2880]).astype("float32") - 0.5
        self.np_shape = [1, 512, 40, 72]
        self.np_dout = np.random.random(size=[1, 512, 40, 72]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase27_FP16(TestReshapeDevelopCase27_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase27_BFP16(TestReshapeDevelopCase27_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase28_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[8]).astype("float32") - 0.5
        self.np_shape = [1, -1, 1, 1]
        self.np_dout = np.random.random(size=[1, 8, 1, 1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase28_FP16(TestReshapeDevelopCase28_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase28_BFP16(TestReshapeDevelopCase28_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase29_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 181, 16, 96]).astype("float32") - 0.5
        self.np_shape = [0, 0, 1536]
        self.np_dout = np.random.random(size=[10, 181, 1536]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase29_FP16(TestReshapeDevelopCase29_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase29_BFP16(TestReshapeDevelopCase29_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase30_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4]).astype("float32") - 0.5
        self.np_shape = [4]
        self.np_dout = np.random.random(size=[4]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase30_FP16(TestReshapeDevelopCase30_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase30_BFP16(TestReshapeDevelopCase30_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase31_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[416, 576, 3]).astype("float32") - 0.5
        self.np_shape = [416, 576, 3]
        self.np_dout = np.random.random(size=[416, 576, 3]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase31_FP16(TestReshapeDevelopCase31_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase31_BFP16(TestReshapeDevelopCase31_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase32_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[576, 352, 3]).astype("float32") - 0.5
        self.np_shape = [576, 352, 3]
        self.np_dout = np.random.random(size=[576, 352, 3]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase32_FP16(TestReshapeDevelopCase32_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase32_BFP16(TestReshapeDevelopCase32_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase33_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 1, 16, 96]).astype("float32") - 0.5
        self.np_shape = [10, 16, 96]
        self.np_dout = np.random.random(size=[10, 16, 96]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase33_FP16(TestReshapeDevelopCase33_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase33_BFP16(TestReshapeDevelopCase33_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase34_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 512, 3, 3]).astype("float32") - 0.5
        self.np_shape = [1179648]
        self.np_dout = np.random.random(size=[1179648]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase34_FP16(TestReshapeDevelopCase34_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase34_BFP16(TestReshapeDevelopCase34_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase35_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 180, 16, 96]).astype("float32") - 0.5
        self.np_shape = [10, 1, 10, 18, 16, 96]
        self.np_dout = np.random.random(size=[10, 1, 10, 18, 16, 96]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase35_FP16(TestReshapeDevelopCase35_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase35_BFP16(TestReshapeDevelopCase35_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase36_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[8, 8, 1, 1]).astype("float32") - 0.5
        self.np_shape = [64]
        self.np_dout = np.random.random(size=[64]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase36_FP16(TestReshapeDevelopCase36_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase36_BFP16(TestReshapeDevelopCase36_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase37_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4, 4, 1, 1]).astype("float32") - 0.5
        self.np_shape = [16]
        self.np_dout = np.random.random(size=[16]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase37_FP16(TestReshapeDevelopCase37_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase37_BFP16(TestReshapeDevelopCase37_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase38_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1536, 1536]).astype("float32") - 0.5
        self.np_shape = [2359296]
        self.np_dout = np.random.random(size=[2359296]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase38_FP16(TestReshapeDevelopCase38_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase38_BFP16(TestReshapeDevelopCase38_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase39_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[3072, 1536]).astype("float32") - 0.5
        self.np_shape = [4718592]
        self.np_dout = np.random.random(size=[4718592]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase39_FP16(TestReshapeDevelopCase39_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase39_BFP16(TestReshapeDevelopCase39_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase40_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 1, 16, 96]).astype("float32") - 0.5
        self.np_shape = [10, 1, 16, 96]
        self.np_dout = np.random.random(size=[10, 1, 16, 96]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase40_FP16(TestReshapeDevelopCase40_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase40_BFP16(TestReshapeDevelopCase40_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase41_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096]).astype("float32") - 0.5
        self.np_shape = [4096]
        self.np_dout = np.random.random(size=[4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase41_FP16(TestReshapeDevelopCase41_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase41_BFP16(TestReshapeDevelopCase41_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase42_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 256, 1536]).astype("float32") - 0.5
        self.np_shape = [0, 0, 16, 96]
        self.np_dout = np.random.random(size=[10, 256, 16, 96]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase42_FP16(TestReshapeDevelopCase42_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase42_BFP16(TestReshapeDevelopCase42_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase43_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4]).astype("float32") - 0.5
        self.np_shape = [1, -1, 1, 1, 1]
        self.np_dout = np.random.random(size=[1, 4, 1, 1, 1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase43_FP16(TestReshapeDevelopCase43_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase43_BFP16(TestReshapeDevelopCase43_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase44_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 2880, 512]).astype("float32") - 0.5
        self.np_shape = [0, 0, 1, 512]
        self.np_dout = np.random.random(size=[1, 2880, 1, 512]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase44_FP16(TestReshapeDevelopCase44_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase44_BFP16(TestReshapeDevelopCase44_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase45_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 10, 32, 128]).astype("float32") - 0.5
        self.np_shape = [256, 10, 4096]
        self.np_dout = np.random.random(size=[256, 10, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase45_FP16(TestReshapeDevelopCase45_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase45_BFP16(TestReshapeDevelopCase45_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase46_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 16384]).astype("float32") - 0.5
        self.np_shape = [67108864]
        self.np_dout = np.random.random(size=[67108864]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase46_FP16(TestReshapeDevelopCase46_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase46_BFP16(TestReshapeDevelopCase46_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase47_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 4, 40, 72]).astype("float32") - 0.5
        self.np_shape = [10, 1, 4, 40, 72]
        self.np_dout = np.random.random(size=[10, 1, 4, 40, 72]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase47_FP16(TestReshapeDevelopCase47_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase47_BFP16(TestReshapeDevelopCase47_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase48_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[320, 256, 256]).astype("float32") - 0.5
        self.np_shape = [10, 32, 256, 256]
        self.np_dout = np.random.random(size=[10, 32, 256, 256]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase48_FP16(TestReshapeDevelopCase48_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase48_BFP16(TestReshapeDevelopCase48_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase49_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[576, 576, 3]).astype("float32") - 0.5
        self.np_shape = [576, 576, 3]
        self.np_dout = np.random.random(size=[576, 576, 3]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase49_FP16(TestReshapeDevelopCase49_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase49_BFP16(TestReshapeDevelopCase49_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase50_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 180, 64]).astype("float32") - 0.5
        self.np_shape = [10, 1, 10, 18, 1, 4, 4, 4]
        self.np_dout = np.random.random(size=[10, 1, 10, 18, 1, 4, 4, 4]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase50_FP16(TestReshapeDevelopCase50_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase50_BFP16(TestReshapeDevelopCase50_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase51_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[320, 256, 128]).astype("float32") - 0.5
        self.np_shape = [10, 32, 256, 128]
        self.np_dout = np.random.random(size=[10, 32, 256, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase51_FP16(TestReshapeDevelopCase51_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase51_BFP16(TestReshapeDevelopCase51_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase52_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1536, 4096]).astype("float32") - 0.5
        self.np_shape = [6291456]
        self.np_dout = np.random.random(size=[6291456]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase52_FP16(TestReshapeDevelopCase52_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase52_BFP16(TestReshapeDevelopCase52_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase53_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 256, 3, 3]).astype("float32") - 0.5
        self.np_shape = [589824]
        self.np_dout = np.random.random(size=[589824]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase53_FP16(TestReshapeDevelopCase53_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase53_BFP16(TestReshapeDevelopCase53_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase54_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1536, 64]).astype("float32") - 0.5
        self.np_shape = [98304]
        self.np_dout = np.random.random(size=[98304]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase54_FP16(TestReshapeDevelopCase54_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase54_BFP16(TestReshapeDevelopCase54_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase55_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[130528, 4096]).astype("float32") - 0.5
        self.np_shape = [534642688]
        self.np_dout = np.random.random(size=[534642688]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase55_FP16(TestReshapeDevelopCase55_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase55_BFP16(TestReshapeDevelopCase55_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase56_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[64]).astype("float32") - 0.5
        self.np_shape = [64]
        self.np_dout = np.random.random(size=[64]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase56_FP16(TestReshapeDevelopCase56_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase56_BFP16(TestReshapeDevelopCase56_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase57_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[128, 256, 1, 1]).astype("float32") - 0.5
        self.np_shape = [32768]
        self.np_dout = np.random.random(size=[32768]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase57_FP16(TestReshapeDevelopCase57_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase57_BFP16(TestReshapeDevelopCase57_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase58_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 32, 256, 256]).astype("float32") - 0.5
        self.np_shape = [320, 256, -1]
        self.np_dout = np.random.random(size=[320, 256, 256]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase58_FP16(TestReshapeDevelopCase58_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase58_BFP16(TestReshapeDevelopCase58_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase59_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 4096]).astype("float32") - 0.5
        self.np_shape = [16777216]
        self.np_dout = np.random.random(size=[16777216]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase59_FP16(TestReshapeDevelopCase59_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase59_BFP16(TestReshapeDevelopCase59_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase60_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[16]).astype("float32") - 0.5
        self.np_shape = [16]
        self.np_dout = np.random.random(size=[16]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase60_FP16(TestReshapeDevelopCase60_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase60_BFP16(TestReshapeDevelopCase60_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase61_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[320, 576, 3]).astype("float32") - 0.5
        self.np_shape = [320, 576, 3]
        self.np_dout = np.random.random(size=[320, 576, 3]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase61_FP16(TestReshapeDevelopCase61_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase61_BFP16(TestReshapeDevelopCase61_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase62_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 4, 1, 1, 10, 4, 18, 4]).astype("float32") - 0.5
        self.np_shape = [10, 4, 1, 40, 72]
        self.np_dout = np.random.random(size=[10, 4, 1, 40, 72]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase62_FP16(TestReshapeDevelopCase62_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase62_BFP16(TestReshapeDevelopCase62_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase63_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 512, 1, 1]).astype("float32") - 0.5
        self.np_shape = [131072]
        self.np_dout = np.random.random(size=[131072]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase63_FP16(TestReshapeDevelopCase63_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase63_BFP16(TestReshapeDevelopCase63_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase64_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[576, 416, 3]).astype("float32") - 0.5
        self.np_shape = [576, 416, 3]
        self.np_dout = np.random.random(size=[576, 416, 3]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase64_FP16(TestReshapeDevelopCase64_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase64_BFP16(TestReshapeDevelopCase64_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase65_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[512, 256, 1, 1]).astype("float32") - 0.5
        self.np_shape = [131072]
        self.np_dout = np.random.random(size=[131072]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase65_FP16(TestReshapeDevelopCase65_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase65_BFP16(TestReshapeDevelopCase65_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase66_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 10, 12288]).astype("float32") - 0.5
        self.np_shape = [256, 10, 32, 384]
        self.np_dout = np.random.random(size=[256, 10, 32, 384]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase66_FP16(TestReshapeDevelopCase66_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase66_BFP16(TestReshapeDevelopCase66_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase67_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[3]).astype("float32") - 0.5
        self.np_shape = [3]
        self.np_dout = np.random.random(size=[3]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase67_FP16(TestReshapeDevelopCase67_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase67_BFP16(TestReshapeDevelopCase67_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase68_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4, 4, 3, 3, 3]).astype("float32") - 0.5
        self.np_shape = [432]
        self.np_dout = np.random.random(size=[432]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase68_FP16(TestReshapeDevelopCase68_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase68_BFP16(TestReshapeDevelopCase68_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase69_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[3, 128, 3, 3]).astype("float32") - 0.5
        self.np_shape = [3456]
        self.np_dout = np.random.random(size=[3456]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase69_FP16(TestReshapeDevelopCase69_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase69_BFP16(TestReshapeDevelopCase69_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase70_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[8]).astype("float32") - 0.5
        self.np_shape = [8]
        self.np_dout = np.random.random(size=[8]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase70_FP16(TestReshapeDevelopCase70_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase70_BFP16(TestReshapeDevelopCase70_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase71_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[128]).astype("float32") - 0.5
        self.np_shape = [128]
        self.np_dout = np.random.random(size=[128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase71_FP16(TestReshapeDevelopCase71_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase71_BFP16(TestReshapeDevelopCase71_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase72_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 128, 1, 1]).astype("float32") - 0.5
        self.np_shape = [32768]
        self.np_dout = np.random.random(size=[32768]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase72_FP16(TestReshapeDevelopCase72_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase72_BFP16(TestReshapeDevelopCase72_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase73_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[3]).astype("float32") - 0.5
        self.np_shape = [-1, 1, 1]
        self.np_dout = np.random.random(size=[3, 1, 1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase73_FP16(TestReshapeDevelopCase73_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase73_BFP16(TestReshapeDevelopCase73_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase74_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[352, 576, 3]).astype("float32") - 0.5
        self.np_shape = [352, 576, 3]
        self.np_dout = np.random.random(size=[352, 576, 3]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase74_FP16(TestReshapeDevelopCase74_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase74_BFP16(TestReshapeDevelopCase74_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase75_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[16384]).astype("float32") - 0.5
        self.np_shape = [16384]
        self.np_dout = np.random.random(size=[16384]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase75_FP16(TestReshapeDevelopCase75_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase75_BFP16(TestReshapeDevelopCase75_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase76_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 1536]).astype("float32") - 0.5
        self.np_shape = [6291456]
        self.np_dout = np.random.random(size=[6291456]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase76_FP16(TestReshapeDevelopCase76_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase76_BFP16(TestReshapeDevelopCase76_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase77_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1536]).astype("float32") - 0.5
        self.np_shape = [1536]
        self.np_dout = np.random.random(size=[1536]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase77_FP16(TestReshapeDevelopCase77_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase77_BFP16(TestReshapeDevelopCase77_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase78_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 10, 32, 128]).astype("float32") - 0.5
        self.np_shape = [256, 320, -1]
        self.np_dout = np.random.random(size=[256, 320, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase78_FP16(TestReshapeDevelopCase78_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase78_BFP16(TestReshapeDevelopCase78_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase79_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1536, 4, 1, 4, 4]).astype("float32") - 0.5
        self.np_shape = [98304]
        self.np_dout = np.random.random(size=[98304]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase79_FP16(TestReshapeDevelopCase79_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase79_BFP16(TestReshapeDevelopCase79_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase80_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[512, 512]).astype("float32") - 0.5
        self.np_shape = [262144]
        self.np_dout = np.random.random(size=[262144]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase80_FP16(TestReshapeDevelopCase80_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase80_BFP16(TestReshapeDevelopCase80_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase81_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 181, 1536]).astype("float32") - 0.5
        self.np_shape = [0, 0, 16, 96]
        self.np_dout = np.random.random(size=[10, 181, 16, 96]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase81_FP16(TestReshapeDevelopCase81_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase81_BFP16(TestReshapeDevelopCase81_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase82_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 1, 3, 320, 576]).astype("float32") - 0.5
        self.np_shape = [10, 3, 320, 576]
        self.np_dout = np.random.random(size=[10, 3, 320, 576]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase82_FP16(TestReshapeDevelopCase82_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase82_BFP16(TestReshapeDevelopCase82_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase83_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[10, 1, 10, 18, 16, 96]).astype("float32") - 0.5
        self.np_shape = [10, 180, 16, 96]
        self.np_dout = np.random.random(size=[10, 180, 16, 96]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase83_FP16(TestReshapeDevelopCase83_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase83_BFP16(TestReshapeDevelopCase83_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase84_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[128, 128, 3, 3]).astype("float32") - 0.5
        self.np_shape = [147456]
        self.np_dout = np.random.random(size=[147456]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase84_FP16(TestReshapeDevelopCase84_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase84_BFP16(TestReshapeDevelopCase84_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase85_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[512, 256, 3, 3]).astype("float32") - 0.5
        self.np_shape = [1179648]
        self.np_dout = np.random.random(size=[1179648]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase85_FP16(TestReshapeDevelopCase85_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase85_BFP16(TestReshapeDevelopCase85_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase86_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1536, 4608]).astype("float32") - 0.5
        self.np_shape = [7077888]
        self.np_dout = np.random.random(size=[7077888]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase86_FP16(TestReshapeDevelopCase86_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase86_BFP16(TestReshapeDevelopCase86_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase87_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[8, 512, 3, 3]).astype("float32") - 0.5
        self.np_shape = [36864]
        self.np_dout = np.random.random(size=[36864]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase87_FP16(TestReshapeDevelopCase87_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase87_BFP16(TestReshapeDevelopCase87_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase88_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256, 128, 3, 3]).astype("float32") - 0.5
        self.np_shape = [294912]
        self.np_dout = np.random.random(size=[294912]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase88_FP16(TestReshapeDevelopCase88_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase88_BFP16(TestReshapeDevelopCase88_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase89_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[512, 4, 3, 3]).astype("float32") - 0.5
        self.np_shape = [18432]
        self.np_dout = np.random.random(size=[18432]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase89_FP16(TestReshapeDevelopCase89_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase89_BFP16(TestReshapeDevelopCase89_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase90_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 12288]).astype("float32") - 0.5
        self.np_shape = [50331648]
        self.np_dout = np.random.random(size=[50331648]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase90_FP16(TestReshapeDevelopCase90_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase90_BFP16(TestReshapeDevelopCase90_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase91_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[16384, 4096]).astype("float32") - 0.5
        self.np_shape = [67108864]
        self.np_dout = np.random.random(size=[67108864]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase91_FP16(TestReshapeDevelopCase91_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase91_BFP16(TestReshapeDevelopCase91_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase92_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[512, 512, 3, 3]).astype("float32") - 0.5
        self.np_shape = [2359296]
        self.np_dout = np.random.random(size=[2359296]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase92_FP16(TestReshapeDevelopCase92_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase92_BFP16(TestReshapeDevelopCase92_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase93_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1536]).astype("float32") - 0.5
        self.np_shape = [1, -1, 1, 1, 1]
        self.np_dout = np.random.random(size=[1, 1536, 1, 1, 1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase93_FP16(TestReshapeDevelopCase93_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase93_BFP16(TestReshapeDevelopCase93_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase94_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 512, 40, 72]).astype("float32") - 0.5
        self.np_shape = [1, 512, 2880]
        self.np_dout = np.random.random(size=[1, 512, 2880]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase94_FP16(TestReshapeDevelopCase94_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase94_BFP16(TestReshapeDevelopCase94_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase95_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[128]).astype("float32") - 0.5
        self.np_shape = [1, -1, 1, 1]
        self.np_dout = np.random.random(size=[1, 128, 1, 1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase95_FP16(TestReshapeDevelopCase95_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase95_BFP16(TestReshapeDevelopCase95_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase96_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[128, 3, 3, 3]).astype("float32") - 0.5
        self.np_shape = [3456]
        self.np_dout = np.random.random(size=[3456]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase96_FP16(TestReshapeDevelopCase96_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase96_BFP16(TestReshapeDevelopCase96_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase97_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1, 2880, 1, 512]).astype("float32") - 0.5
        self.np_shape = [0, 0, 512]
        self.np_dout = np.random.random(size=[1, 2880, 512]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase97_FP16(TestReshapeDevelopCase97_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase97_BFP16(TestReshapeDevelopCase97_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestReshapeDevelopCase98_FP32(TestReshapeDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[256]).astype("float32") - 0.5
        self.np_shape = [256]
        self.np_dout = np.random.random(size=[256]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestReshapeDevelopCase98_FP16(TestReshapeDevelopCase98_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestReshapeDevelopCase98_BFP16(TestReshapeDevelopCase98_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()