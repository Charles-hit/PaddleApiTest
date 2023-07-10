import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("../..")
from utils import TOLERANCE, convert_dtype_to_torch_type, np_assert_accuracy


niuliling_path = None  # 全局变量
class TestLayerNormFP32vsBFP16(unittest.TestCase):
    def setUp(self):
        self.init_np_inputs_and_dout()

    def init_np_inputs_and_dout(self):
        # init np array 
        data_xwb = np.load(niuliling_path+".npz")
        data_dout = np.load(niuliling_path+".npy")
        self.np_x = data_xwb["x"].astype("float32")
        self.np_w = data_xwb["weight"].astype("float32")
        self.np_b = data_xwb["bias"].astype("float32")
        self.np_dout = data_dout.astype("float32")
    
    def gen_eager_inputs_and_dout(self):
        x = paddle.to_tensor(
            self.np_x,
            dtype="float32",
            place="gpu",
        )
        x.stop_gradient = False
        w = paddle.to_tensor(
            self.np_w,
            dtype="float32",
            place="gpu",
        )
        w.stop_gradient = False
        b = paddle.to_tensor(
            self.np_b,
            dtype="float32",
            place="gpu",
        )
        b.stop_gradient = False
        dout = paddle.to_tensor(
            self.np_dout,
            dtype="float32",
            place="gpu",
        )
        dout.stop_gradient = False
        return x, w, b, dout

    def cal_res(self, x, w, b, dout):
        out = paddle.nn.functional.layer_norm(x, x.shape[2:], w, b, 1e-05)
        out_grads = paddle.grad([out], [x, w, b], grad_outputs=[dout])
        return out, out_grads


    def test_layernorm_fp32vsbfp16_mode1(self):
        x_bfp16, w_bfp16, b_bfp16, dout_bfp16 = map_structure(lambda x: paddle.cast(x, dtype="bfloat16"), self.gen_eager_inputs_and_dout())
        x_fp32, w_fp32, b_fp32, dout_fp32 = paddle.cast(x_bfp16,"float32"), paddle.cast(w_bfp16,"float32"), paddle.cast(b_bfp16,"float32"), paddle.cast(dout_bfp16,"float32")
        out_fp32, out_grads_fp32 = self.cal_res(x_fp32, w_fp32, b_fp32, dout_fp32)
        out_bfp16, out_grads_bfp16 = self.cal_res(x_bfp16, w_bfp16, b_bfp16, dout_bfp16)
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
                api="paddle.nn.functional.layernorm",
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
                    api="paddle.nn.functional.layernorm",
                )
        except Exception as e:
            print(e)

    def test_layernorm_fp32vsbfp16_mode2(self):
        x_bfp16, w_bfp16, b_bfp16, dout_bfp16 = map_structure(lambda x: paddle.cast(x, dtype="bfloat16"), self.gen_eager_inputs_and_dout())
        x_fp32, w_fp32, b_fp32, dout_fp32 = paddle.cast(x_bfp16,"float32"), paddle.cast(w_bfp16,"float32"), paddle.cast(b_bfp16,"float32"), paddle.cast(dout_bfp16,"float32")
        out_fp32, out_grads_fp32 = self.cal_res(x_fp32, w_fp32, b_fp32, dout_fp32)
        out_fp32 = paddle.cast(paddle.cast(out_fp32, "bfloat16"), "float32")
        out_grads_fp32 =  map_structure(lambda x: paddle.cast(paddle.cast(x,"bfloat16"),"float32"), out_grads_fp32)
        out_bfp16, out_grads_bfp16 = self.cal_res(x_bfp16, w_bfp16, b_bfp16, dout_bfp16)
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
                eager_or_static_mode="mode1",
                fwd_or_bkd="forward",
                api="paddle.nn.functional.layernorm",
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
                    eager_or_static_mode="mode1",
                    fwd_or_bkd="backward",
                    api="paddle.nn.functional.layernorm",

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
