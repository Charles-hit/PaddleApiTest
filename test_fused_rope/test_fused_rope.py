#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import torchtune

import paddle
from paddle.utils import map_structure
from paddle.incubate.nn.functional import fused_rotary_position_embedding

sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)


class TestFusedRotaryPositionEmbedding_FP32_Case0(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        q_torch, k_torch, v_torch, dout_q_torch, dout_k_torch, dout_v_torch = (
            self.gen_torch_inputs_and_dout()
        )
        outs_torch, out_grads_torch = self.cal_torch_res(
            q_torch, k_torch, v_torch, dout_q_torch, dout_k_torch, dout_v_torch
        )
        del q_torch
        del k_torch
        del v_torch
        del dout_q_torch
        del dout_k_torch
        del dout_v_torch
        self.outs_torch = map_structure(
            lambda x: x.detach().cpu().numpy(),
            outs_torch,
        )
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del outs_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.shape_q = [1, 8192, 12, 128]
        self.shape_k = None
        self.shape_v = None

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array
        self.np_q = np.random.random(size=self.shape_q).astype("float32") - 0.5
        self.np_dout_q = np.random.random(size=self.shape_q).astype("float32") - 0.5
        if self.shape_k is not None:
            self.np_k = np.random.random(size=self.shape_k).astype("float32") - 0.5
            self.np_dout_k = np.random.random(size=self.shape_k).astype("float32") - 0.5
        if self.shape_v is not None:
            self.np_v = np.random.random(size=self.shape_v).astype("float32") - 0.5
            self.np_dout_v = np.random.random(size=self.shape_k).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_q = self.np_q.astype("float16")
            self.np_dout_q = self.np_dout_q.astype("float16")
            if self.shape_v is not None:
                self.np_v = self.np_v.astype("float16")
                self.np_dout_v = self.np_dout_v.astype("float16")
            if self.shape_k is not None:
                self.np_k = self.np_k.astype("float16")
                self.np_dout_k = self.np_dout_k.astype("float16")

    def gen_torch_inputs_and_dout(self):
        q_torch = torch.tensor(
            self.np_q,
            device="cuda",
            dtype=(
                convert_dtype_to_torch_type(self.dtype)
                if self.dtype != "bfloat16"
                else torch.float32
            ),
            requires_grad=True,
        )
        dout_q_torch = torch.tensor(
            self.np_dout_q,
            device="cuda",
            dtype=(
                convert_dtype_to_torch_type(self.dtype)
                if self.dtype != "bfloat16"
                else torch.float32
            ),
            requires_grad=True,
        )
        if self.shape_k is not None:
            k_torch = torch.tensor(
                self.np_k,
                device="cuda",
                dtype=(
                    convert_dtype_to_torch_type(self.dtype)
                    if self.dtype != "bfloat16"
                    else torch.float32
                ),
                requires_grad=True,
            )
            dout_k_torch = torch.tensor(
                self.np_dout_k,
                device="cuda",
                dtype=(
                    convert_dtype_to_torch_type(self.dtype)
                    if self.dtype != "bfloat16"
                    else torch.float32
                ),
                requires_grad=True,
            )
        else:
            k_torch = None
            dout_k_torch = None
        if self.shape_v is not None:
            v_torch = torch.tensor(
                self.np_v,
                device="cuda",
                dtype=(
                    convert_dtype_to_torch_type(self.dtype)
                    if self.dtype != "bfloat16"
                    else torch.float32
                ),
                requires_grad=True,
            )
            dout_v_torch = torch.tensor(
                self.np_dout_v,
                device="cuda",
                dtype=(
                    convert_dtype_to_torch_type(self.dtype)
                    if self.dtype != "bfloat16"
                    else torch.float32
                ),
                requires_grad=True,
            )
        else:
            v_torch = None
            dout_v_torch = None

        return q_torch, k_torch, v_torch, dout_q_torch, dout_k_torch, dout_v_torch

    def gen_eager_inputs_and_dout(self):
        q_eager = paddle.to_tensor(
            self.np_q,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            place="gpu",
        )
        q_eager.stop_gradient = False
        dout_q_eager = paddle.to_tensor(
            self.np_dout_q,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            place="gpu",
        )
        dout_q_eager.stop_gradient = False
        if self.shape_k is not None:
            k_eager = paddle.to_tensor(
                self.np_k,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
                place="gpu",
            )
            k_eager.stop_gradient = False
            dout_k_eager = paddle.to_tensor(
                self.np_dout_k,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
                place="gpu",
            )
            dout_k_eager.stop_gradient = False
        else:
            k_eager = None
            dout_k_eager = None
        if self.shape_v is not None:
            v_eager = paddle.to_tensor(
                self.np_v,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
                place="gpu",
            )
            v_eager.stop_gradient = False
            dout_v_eager = paddle.to_tensor(
                self.np_dout_v,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
                place="gpu",
            )
            dout_v_eager.stop_gradient = False
        else:
            v_eager = None
            dout_v_eager = None

        return q_eager, k_eager, v_eager, dout_q_eager, dout_k_eager, dout_v_eager

    def cal_eager_res(self, q, k, v, dout_q, dout_k, dout_v):
        if self.dtype == "bfloat16":
            q = paddle.cast(q, dtype="uint16")
            k = paddle.cast(k, dtype="uint16") if k is not None else None
            v = paddle.cast(v, dtype="uint16") if v is not None else None
            dout_q = paddle.cast(dout_q, dtype="uint16")
            dout_k = paddle.cast(dout_k, dtype="uint16") if dout_k is not None else None
            dout_v = paddle.cast(dout_v, dtype="uint16") if dout_v is not None else None
        out_q, out_k, out_v = fused_rotary_position_embedding(q, k, v)
        outs = [out_q, out_k, out_v]
        outs = list(filter(lambda x: x._is_initialized(), outs))
        grad_outputs = list(filter(lambda x: x is not None, [dout_q, dout_k, dout_v]))
        inputs = list(filter(lambda x: x is not None, [q, k, v]))
        out_grads = paddle.grad(outs, inputs, grad_outputs=grad_outputs)
        if self.dtype == "bfloat16":
            outs = map_structure(lambda x: paddle.cast(x, dtype="float32"), outs)
            out_grads = map_structure(
                lambda x: paddle.cast(x, dtype="float32"), out_grads
            )
        return outs, out_grads

    def cal_torch_res(self, q, k, v, dout_q, dout_k, dout_v):
        if self.dtype == "bfloat16":
            q = q.to(dtype=torch.bfloat16)
            k = k.to(dtype=torch.bfloat16) if k is not None else None
            v = v.to(dtype=torch.bfloat16) if v is not None else None
            dout_q = dout_q.to(dtype=torch.bfloat16) if dout_q is not None else None
            dout_k = dout_k.to(dtype=torch.bfloat16) if dout_k is not None else None
            dout_v = dout_v.to(dtype=torch.bfloat16) if dout_v is not None else None
        rope_q = torchtune.modules.RotaryPositionalEmbeddings(
            q.shape[-1], q.shape[1]
        ).to("cuda")
        rope_k = (
            torchtune.modules.RotaryPositionalEmbeddings(k.shape[-1], k.shape[1]).to(
                "cuda"
            )
            if k is not None
            else None
        )
        rope_v = (
            torchtune.modules.RotaryPositionalEmbeddings(v.shape[-1], v.shape[1]).to(
                "cuda"
            )
            if v is not None
            else None
        )
        rope_q.reset_parameters()
        rope_k.reset_parameters() if rope_k is not None else None
        rope_v.reset_parameters() if rope_v is not None else None
        rope_q.to("cuda")
        rope_k.to("cuda") if rope_k is not None else None
        rope_v.to("cuda") if rope_v is not None else None
        out_q = rope_q(q)
        out_k = rope_k(k) if k is not None else None
        out_v = rope_v(v) if v is not None else None
        outs = [out_q, out_k, out_v]
        outs = list(filter(lambda x: x is not None, outs))
        grad_outputs = list(filter(lambda x: x is not None, [dout_q, dout_k, dout_v]))
        inputs = list(filter(lambda x: x is not None, [q, k, v]))
        out_grads = torch.autograd.grad(
            outs,
            inputs,
            grad_outputs=grad_outputs,
        )
        if self.dtype == "bfloat16":
            outs = map_structure(lambda x: x.to(dtype=torch.float32), outs)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return outs, out_grads

    def test_eager_accuracy(self):
        paddle.disable_static()
        q_eager, k_eager, v_eager, dout_q_eager, dout_k_eager, dout_v_eager = (
            self.gen_eager_inputs_and_dout()
        )
        outs_eager, out_grads_eager = self.cal_eager_res(
            q_eager, k_eager, v_eager, dout_q_eager, dout_k_eager, dout_v_eager
        )

        del q_eager
        del k_eager
        del v_eager
        del dout_q_eager
        del dout_k_eager
        del dout_v_eager
        paddle.device.cuda.empty_cache()
        outs_eager_np = map_structure(
            lambda x: x.numpy(),
            outs_eager,
        )
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        del outs_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()
        # compare develop eager forward res with torch
        for idx in range(len(outs_eager_np)):
            np_assert_accuracy(
                outs_eager_np[idx],
                self.outs_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="fused_rotary_position_embedding",
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
                api="fused_rotary_position_embedding",
            )


class TestFusedRotaryPositionEmbedding_FP32_Case1(
    TestFusedRotaryPositionEmbedding_FP32_Case0
):
    def init_params(self):
        self.dtype = "float32"
        self.shape_q = [1, 8192, 1, 128]
        self.shape_k = None
        self.shape_v = None


class TestFusedRotaryPositionEmbedding_FP16_Case0(
    TestFusedRotaryPositionEmbedding_FP32_Case0
):
    def init_params(self):
        self.dtype = "float16"
        self.shape_q = [1, 8192, 12, 128]
        self.shape_k = None
        self.shape_v = None


class TestFusedRotaryPositionEmbedding_FP16_Case1(
    TestFusedRotaryPositionEmbedding_FP32_Case0
):
    def init_params(self):
        self.dtype = "float32"
        self.shape_q = [1, 8192, 1, 128]
        self.shape_k = None
        self.shape_v = None


class TestFusedRotaryPositionEmbedding_FP16_Case1(
    TestFusedRotaryPositionEmbedding_FP32_Case0
):
    def init_params(self):
        self.dtype = "float16"
        self.shape_q = [1, 8192, 1, 128]
        self.shape_k = None
        self.shape_v = None


class TestFusedRotaryPositionEmbedding_BF16_Case0(
    TestFusedRotaryPositionEmbedding_FP32_Case0
):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_q = [1, 8192, 12, 128]
        self.shape_k = None
        self.shape_v = None


class TestFusedRotaryPositionEmbedding_BF16_Case1(
    TestFusedRotaryPositionEmbedding_FP32_Case0
):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_q = [1, 8192, 1, 128]
        self.shape_k = None
        self.shape_v = None


if __name__ == "__main__":
    np.random.seed(2024)
    unittest.main()
