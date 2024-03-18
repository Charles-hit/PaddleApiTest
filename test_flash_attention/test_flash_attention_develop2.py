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
from paddle.nn.functional.flash_attention import flash_attention

sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

class TestFlashAttentionDevelopCase1_FP16(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        q_torch, k_torch, v_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            q_torch, k_torch, v_torch, dout_torch
        )
        del q_torch
        del k_torch
        del v_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float16"
        self.shape_q = [10, 181, 16, 96]
        self.shape_k = [10, 256, 16, 96]
        self.shape_v = [10, 256, 16, 96]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_q = np.random.random(size=self.shape_q).astype("float32") - 0.5
        self.np_k = np.random.random(size=self.shape_k).astype("float32") - 0.5
        self.np_v = np.random.random(size=self.shape_v).astype("float32") - 0.5
        self.np_dout = np.random.random(size=self.shape_q).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_q = self.np_q.astype("float16")
            self.np_v = self.np_v.astype("float16")
            self.np_k = self.np_k.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def gen_torch_inputs_and_dout(self):
        q_torch = torch.tensor(
            self.np_q,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        k_torch = torch.tensor(
            self.np_k,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        v_torch = torch.tensor(
            self.np_v,
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
        return q_torch,k_torch,v_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        q_eager = paddle.to_tensor(
            self.np_q,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        q_eager.stop_gradient = False
        k_eager = paddle.to_tensor(
            self.np_k,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        k_eager.stop_gradient = False
        v_eager = paddle.to_tensor(
            self.np_v,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        v_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return q_eager,k_eager,v_eager, dout_eager

    def cal_torch_res(self, q,k,v, dout):
        q_t = torch.permute(q, (0, 2, 1, 3))
        k_t = torch.permute(k, (0, 2, 1, 3))
        v_t = torch.permute(v, (0, 2, 1, 3))

        if self.dtype == "bfloat16":
            q_t = q_t.to(dtype=torch.bfloat16)
            k_t = k_t.to(dtype=torch.bfloat16)
            v_t = v_t.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)

        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
        out = torch.permute(out, (0, 2, 1, 3))
        out_grads = torch.autograd.grad([out], [q_t,k_t,v_t], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        out_grads = [out_grad.permute(0, 2, 1, 3) for out_grad in out_grads]
        return out, out_grads

    def cal_eager_res(self, q, k, v, dout):
        if self.dtype == "bfloat16":
            q = paddle.cast(q, dtype="uint16")
            k = paddle.cast(k, dtype="uint16")
            v = paddle.cast(v, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out, _ = flash_attention(q, k, v, 0.0, False, False)
        out_grads = paddle.grad([out], [q,k,v], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads


    def test_eager_accuracy(self):
        q_eager,k_eager,v_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            q_eager,k_eager,v_eager, dout_eager
        )

        del q_eager
        del k_eager
        del v_eager
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
            api="flash_attention",
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
                api="flash_attention",
            )


class TestFlashAttentionDevelopCase1_BFP16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_q = [10, 181, 16, 96]
        self.shape_k = [10, 256, 16, 96]
        self.shape_v = [10, 256, 16, 96]

class TestFlashAttentionDevelopCase2_FP16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "float16"
        self.shape_q = [1, 1, 2, 40]
        self.shape_k = [1, 1, 2, 40]
        self.shape_v = [1, 1, 2, 40]

class TestFlashAttentionDevelopCase2_BF16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_q = [1, 1, 2, 40]
        self.shape_k = [1, 1, 2, 40]
        self.shape_v = [1, 1, 2, 40]


class TestFlashAttentionDevelopCase3_FP16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "float16"
        self.shape_q = [10, 1, 16, 96]
        self.shape_k = [10, 181, 16, 96]
        self.shape_v = [10, 181, 16, 96]
class TestFlashAttentionDevelopCase3_BF16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_q = [10, 1, 16, 96]
        self.shape_k = [10, 181, 16, 96]
        self.shape_v = [10, 181, 16, 96]


class TestFlashAttentionDevelopCase4_FP16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "float16"
        self.shape_q = [1, 2880, 1, 512]
        self.shape_k = [1, 2880, 1, 512]
        self.shape_v = [1, 2880, 1, 512]
class TestFlashAttentionDevelopCase4_BF16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_q = [1, 2880, 1, 512]
        self.shape_k = [1, 2880, 1, 512]
        self.shape_v = [1, 2880, 1, 512]

class TestFlashAttentionDevelopCase5_FP16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "float16"
        self.shape_q = [10, 180, 16, 96]
        self.shape_k = [10, 181, 16, 96]
        self.shape_v = [10, 181, 16, 96]
class TestFlashAttentionDevelopCase5_BF16(TestFlashAttentionDevelopCase1_FP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_q = [10, 180, 16, 96]
        self.shape_k = [10, 181, 16, 96]
        self.shape_v = [10, 181, 16, 96]

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
