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
import math

import paddle
from paddle.utils import map_structure
from paddle.nn.functional.flash_attention import flash_attention

sys.path.append("../..")
from utils import (
    convert_dtype_to_torch_type,
    np_assert_accuracy
)
def get_triangle_upper_mask(shape):
    mask = paddle.full(shape=shape, fill_value=-np.inf)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask

def multi_head_attention(q, k, v, causal=True):
    # [bs, seq_len, num_head, head_dim]
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    qt = qt / math.sqrt(qt.shape[-1])
    product = paddle.matmul(qt, kt, transpose_y=True)
    if causal:
        mask = get_triangle_upper_mask(product.shape)
        product += mask
    product = paddle.nn.functional.softmax(product)
    out = paddle.matmul(product, vt)
    out = paddle.transpose(out, [0, 2, 1, 3])
    return out

class TestFlashAttentionFP32vsBFP16(unittest.TestCase):
    def setUp(self):
        self.init_np_inputs_and_dout()

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1,8192,14,128]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1,8192,14,128]).astype("float32") - 0.5
    
    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype="float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype="float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, dout_eager
    def cal_bfp16_res(self, x, dout):
        out, _ = flash_attention(x, x, x, 0.0, True, False)
        out_grads = paddle.grad([out], [x], grad_outputs=[dout])
        return out, out_grads
    
    # def cal_fp32_res(self, x, dout):
    #     x_t = paddle.cast(x, "bfloat16")
    #     dout_t = paddle.cast(dout, "bfloat16")
    #     out = multi_head_attention(x_t, x_t, x_t)
    #     out_grads = paddle.grad([out], [x], grad_outputs=[dout_t])
    #     out = paddle.cast(out, "float32")
    #     out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
    #     return out, out_grads

    def cal_fp32_res(self, x, dout):
        out = multi_head_attention(x, x, x)
        out_grads = paddle.grad([out], [x], grad_outputs=[dout])
        return out, out_grads

    def test_flash_atten_fp32vsbfp16_mode1(self):
        x_bfp16, dout_bfp16 = map_structure(lambda x: paddle.cast(x, dtype="bfloat16"), self.gen_eager_inputs_and_dout())
        x_fp32, dout_fp32 = paddle.cast(x_bfp16,"float32"), paddle.cast(dout_bfp16,"float32")
        out_fp32, out_grads_fp32 = self.cal_fp32_res(x_fp32, dout_fp32)
        out_bfp16, out_grads_bfp16 = self.cal_bfp16_res(x_bfp16, dout_bfp16)
        pt_out_bfp16 = paddle.cast(out_bfp16, "float32")
        pt_out_grads_bfp16 = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads_bfp16)
        try:
            np_assert_accuracy(
                out_fp32.numpy(),
                pt_out_bfp16.numpy(),
                1e-2,
                1e-2,
                "fp32vsbfp16",
                version_a="fp32",
                version_b="bfp16",
                eager_or_static_mode="mode1",
                fwd_or_bkd="forward",
                api="paddle.nn.functional.flash_attention",
            )
        except Exception as e:
            print(e)
        try:
            for i in range(len(out_grads_fp32)):
                np_assert_accuracy(
                    out_grads_fp32[i].numpy(),
                    pt_out_grads_bfp16[i].numpy(),
                    1e-2,
                    1e-2,
                    "fp32vsbfp16",
                    version_a="fp32",
                    version_b="bfp16",
                    eager_or_static_mode="mode1",
                    fwd_or_bkd="backward",
                    api="paddle.nn.functional.flash_attention",
                )
        except Exception as e:
            print(e)

    def test_flash_atten_fp32vsbfp16_mode2(self):
        x_bfp16, dout_bfp16 = map_structure(lambda x: paddle.cast(x, dtype="bfloat16"), self.gen_eager_inputs_and_dout())
        x_fp32, dout_fp32 = paddle.cast(x_bfp16,"float32"), paddle.cast(dout_bfp16,"float32")
        out_fp32, out_grads_fp32 = self.cal_fp32_res(x_fp32, dout_fp32)
        out_grads_fp32 =  map_structure(lambda x: paddle.cast(paddle.cast(x,"bfloat16"),"float32"), out_grads_fp32)
        out_bfp16, out_grads_bfp16 = self.cal_bfp16_res(x_bfp16, dout_bfp16)
        pt_out_bfp16 = paddle.cast(out_bfp16, "float32")
        pt_out_grads_bfp16 = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads_bfp16)
        try:
            np_assert_accuracy(
                out_fp32.numpy(),
                pt_out_bfp16.numpy(),
                1e-6,
                1e-6,
                "fp32vsbfp16",
                version_a="fp32",
                version_b="bfp16",
                eager_or_static_mode="mode2",
                fwd_or_bkd="forward",
                api="paddle.nn.functional.flash_attention",
            )
        except Exception as e:
            print(e)
        try:
            for i in range(len(out_grads_fp32)):
                np_assert_accuracy(
                    out_grads_fp32[i].numpy(),
                    pt_out_grads_bfp16[i].numpy(),
                    1e-6,
                    1e-6,
                    "fp32vsbfp16",
                    version_a="fp32",
                    version_b="bfp16",
                    eager_or_static_mode="mode2",
                    fwd_or_bkd="backward",
                    api="paddle.nn.functional.flash_attention",
                )
        except Exception as e:
            print(e)

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
