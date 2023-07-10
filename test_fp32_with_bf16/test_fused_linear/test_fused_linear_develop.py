
import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("../..")
from utils import (
    TOLERANCE,
    np_assert_accuracy
)

global_out = []
global_dout = []
niuliling_path = None  # 全局变量

class TestFCDevelop(unittest.TestCase):
    def __init__(self, shape, dtype, test_mode):
        self.dtype = dtype
        self.shape = shape  
        self.init_threshold()
        self.init_np_inputs_and_dout()
        self.test_mode = test_mode
    
    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np.random.seed(123)
        data_xwb = np.load(niuliling_path+".npz")
        data_dout = np.load(niuliling_path+".npy")
        self.np_x = data_xwb["x"]
        self.np_w = data_xwb["weight"]
        self.np_b = data_xwb["bias"]
        print("shape x ", self.np_x.shape)
        print("shape w ", self.np_w.shape)
        print("shape b ", self.np_b.shape)
        self.np_dout = data_dout.astype("float32")

        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_w = self.np_w.astype("float16")
            self.np_b = self.np_b.astype("float16")
            self.np_dout = self.np_dout.astype("float16")   
    
    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        w_eager = paddle.to_tensor(
            self.np_w,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        w_eager.stop_gradient = False
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
        return x_eager, w_eager, b_eager, dout_eager

    def cal_eager_res(self, x, w, b, dout):
        x_t = x
        w_t = w
        b_t = b
        dout_t = dout

        if self.dtype == "float32":
            x_t = paddle.cast(x, dtype="uint16")
            w_t = paddle.cast(w, dtype="uint16")
            b_t = paddle.cast(b, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")

            x_t = paddle.cast(x, dtype="float32")
            w_t = paddle.cast(w, dtype="float32")
            b_t = paddle.cast(b, dtype="float32")
            dout_t = paddle.cast(dout, dtype="float32")

        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            w_t = paddle.cast(w, dtype="uint16")
            b_t = paddle.cast(b, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.incubate.nn.functional.fused_linear(x_t, w_t, b_t)
        out_grads = paddle.grad(
            [out], [x, w, b], grad_outputs=[dout_t]
        )
        out_grads = out_grads[0]
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")

        if self.test_mode == 2 and self.dtype == "float32":
            out = paddle.cast(out, dtype="uint16")
            out = paddle.cast(out, dtype="float32")
            out_grads = paddle.cast(out_grads, dtype="uint16")
            out_grads = paddle.cast(out_grads, dtype="float32")

        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, w_eager, b_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, w_eager, b_eager, dout_eager)
        del x_eager
        del w_eager 
        del b_eager
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = out_grads_eager.numpy()
        global_out.append(out_eager_np)
        global_dout.append(out_grads_eager_np)

        del out_eager
        del out_grads_eager

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("请提供 data_path 参数")
        sys.exit(1)

    tmp = sys.argv[1]  # 设置全局变量 data_path
    niuliling_path = tmp
    print(tmp) 
    del sys.argv[1]


    x = [[1, 8192, 14336]] #, [1, 8192, 14336], [1, 8192, 1792], [1, 8192, 4816]]
    w = [[14336, 5376]] #, [14336, 9632], [1792, 14336], [4816, 14336]]
    b = [[5376], [9632]] #, [14336], [14336]]
    dout = [[1, 8192, 5376]]#, [1, 8192, 9632], [1, 8192, 14336], [1, 8192, 14336]]
    shape_list = []
    for i in range(len(x)):
        shape = {}
        shape["x"] = x[i]
        shape["w"] = w[i]
        shape["b"] = b[i]
        shape["dout"] = dout[i]
        shape_list.append(shape)

    for test_mode in [1,2]:
        if test_mode == 1:
            atol = 1e-2
        elif test_mode == 2:
            atol = 1e-6
        #print("test_mode_{test_mode} start*************************************************************************" \
        #    .format(test_mode=test_mode))
        print(tmp, "test_mode", test_mode) 
        for shape in shape_list:
            global_out.clear()
            global_dout.clear()
            print("shape: ", shape)
            test_case_fp32 = TestFCDevelop(shape, dtype="float32", test_mode=test_mode)
            test_case_fp32.test_eager_accuracy()
            test_case_bf16 = TestFCDevelop(shape, dtype="bfloat16", test_mode=test_mode)
            test_case_bf16.test_eager_accuracy()
            try:
                np_assert_accuracy(
                    global_out[0],
                    global_out[1],
                    atol,
                    atol,
                    "fp32_vs_bf16",
                    version_a="fp32",
                    version_b="bf16",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="paddle.incubate.nn.functional.fused_linear",
                )
            except Exception as e:
                print(e)

            try:
                print("global_dout[0].shape: ", global_dout[0].shape)
                print("global_dout[1].shape: ", global_dout[1].shape)
                np_assert_accuracy(
                    global_dout[0],
                    global_dout[1],
                    atol,
                    atol,
                    "fp32_vs_bf16",
                    version_a="fp32",
                    version_b="bf16",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="paddle.incubate.nn.functional.fused_linear",
                )
            except Exception as e:
                print(e)
        print("test_mode_{test_mode} end*************************************************************************" \
            .format(test_mode=test_mode))
