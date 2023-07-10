
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

sys.path.append("../..")
from utils import (
    np_assert_accuracy,
)
niuliling_path = None  # 全局变量

class TestMatmulFP32vsBFP16(unittest.TestCase):
    def setUp(self):
        self.init_np_inputs_and_dout()

    def init_np_inputs_and_dout(self):
        # init np array 
        data_xy = np.load(niuliling_path+".npz")
        data_dout = np.load(niuliling_path+".npy")

        self.np_x = data_xy["x"].astype("float32")
        self.np_y = data_xy["y"].astype("float32")
        print("shape x", self.np_x.shape)
        print("shape y", self.np_y.shape)
        self.np_dout = data_dout.astype("float32")
    
    def gen_eager_inputs_and_dout(self):
        x = paddle.to_tensor(
            self.np_x,
            dtype="float32",
            place="gpu",
        )
        x.stop_gradient = False
        y = paddle.to_tensor(
            self.np_y,
            dtype="float32",
            place="gpu",
        )
        y.stop_gradient = False
        dout = paddle.to_tensor(
            self.np_dout,
            dtype="float32",
            place="gpu",
        )
        dout.stop_gradient = False
        return x, y, dout

    def cal_res(self, x, y, dout):
        out = paddle.matmul(x, y)
        out_grads = paddle.grad([out], [x, y], grad_outputs=[dout])
        return out, out_grads


    def test_matmul_fp32vsbfp16_mode1(self):
        x_bfp16, y_bfp16, dout_bfp16 = map_structure(lambda x: paddle.cast(x, dtype="bfloat16"), self.gen_eager_inputs_and_dout())
        x_fp32, y_fp32, dout_fp32 = paddle.cast(x_bfp16,"float32"), paddle.cast(y_bfp16,"float32"), paddle.cast(dout_bfp16,"float32")
        out_fp32, out_grads_fp32 = self.cal_res(x_fp32, y_fp32, dout_fp32)
        out_bfp16, out_grads_bfp16 = self.cal_res(x_bfp16, y_bfp16, dout_bfp16)
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
                api="paddle.matmul",
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
                    api="paddle.matmul",
                )
        except Exception as e:
            print(e)

    def test_matmul_fp32vsbfp16_mode2(self):
        x_bfp16, y_bfp16, dout_bfp16 = map_structure(lambda x: paddle.cast(x, dtype="bfloat16"), self.gen_eager_inputs_and_dout())
        x_fp32, y_fp32, dout_fp32 = paddle.cast(x_bfp16,"float32"), paddle.cast(y_bfp16,"float32"), paddle.cast(dout_bfp16,"float32")
        out_fp32, out_grads_fp32 = self.cal_res(x_fp32, y_fp32, dout_fp32)
        out_fp32 = paddle.cast(paddle.cast(out_fp32,"bfloat16"),"float32")
        out_grads_fp32 =  map_structure(lambda x: paddle.cast(paddle.cast(x,"bfloat16"),"float32"), out_grads_fp32)
        out_bfp16, out_grads_bfp16 = self.cal_res(x_bfp16, y_bfp16, dout_bfp16)
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
                api="paddle.matmul",
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
                    api="paddle.matmul",
                )
        except Exception as e:
            print(e)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("请提供 data_path 参数")
        sys.exit(1)

    tmp = sys.argv[1]  # 设置全局变量 data_path
    niuliling_path = tmp
    print(tmp) 
    del sys.argv[1]
    np.random.seed(2023)
    unittest.main()
