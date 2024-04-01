import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("..")
sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

class TestLinearDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, weight_torch, dout_torch , bias_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(x_torch, weight_torch, dout_torch , bias_torch)
        del x_torch
        del weight_torch
        del dout_torch
        del bias_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
                                lambda x: x.cpu().detach().numpy(),
                                out_grads_torch,
                            )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.shape_x = [256, 10, 4096]
        self.shape_w = [4096, 12288]
        self.shape_b = [12288]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        self.np_x = (np.random.random(size=self.shape_x).astype("float32") - 0.5).astype("float32")
        self.np_weight = (np.random.random(size=self.shape_w).astype("float32")-0.5).astype("float32")
        if self.shape_b != None:
            self.np_bias = ((np.random.random(size=self.shape_b)).astype("float32")-0.5).astype("float32")
            self.np_dout = (np.random.random(size=paddle.nn.functional.linear(paddle.to_tensor(self.np_x), paddle.to_tensor(self.np_weight), paddle.to_tensor(self.np_bias)).shape).astype("float32")-0.5).astype("float32")
        else:
            self.np_dout = (np.random.random(size=paddle.nn.functional.linear(paddle.to_tensor(self.np_x), paddle.to_tensor(self.np_weight), None).shape).astype("float32")-0.5).astype("float32")
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_weight = self.np_weight.astype("float16")
            self.np_dout = self.np_dout.astype("float16")
            if self.shape_b != None:
                self.np_bias = self.np_bias.astype("float16")


    def gen_torch_inputs_and_dout(self):
        the_device='cuda'
        x_torch = torch.tensor(
            self.np_x,
            device=the_device,
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        weight_torch = torch.tensor(
            self.np_weight,
            device=the_device,
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dout_torch = torch.tensor(
            self.np_dout,
            device=the_device,
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        if self.shape_b != None:
            bias_torch = torch.tensor(
                self.np_bias,
                device=the_device,
                dtype=convert_dtype_to_torch_type(self.dtype)
                if self.dtype != 'bfloat16'
                else torch.float32,
                requires_grad=True,
            )
        else:
            bias_torch = None

        return x_torch, weight_torch, dout_torch , bias_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        weight_eager = paddle.to_tensor(
            self.np_weight,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        weight_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        if self.shape_b != None:
            bias_eager = paddle.to_tensor(
                self.np_bias,
                dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
                place="gpu",
            )
            bias_eager.stop_gradient = False
        else:
            bias_eager=None
        return x_eager, weight_eager, dout_eager , bias_eager

    def cal_torch_res(self, x, weight, dout, bias):
        bias = None
        x_t = x
        weight_t = weight
        dout_t = dout
        bias_t = bias
        if self.dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            weight_t = weight.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
            if bias != None:
                bias_t = bias.to(dtype=torch.bfloat16)
            else:
                bias_t = bias

        weight_t=torch.transpose(weight_t ,dim0=0, dim1=1)
        out = torch.nn.functional.linear(x_t, weight_t , bias_t)
        if bias is not None:
            out_grads = torch.autograd.grad([out], [x, weight, bias], grad_outputs=[dout_t])
        else:
            out_grads = torch.autograd.grad([out], [x, weight], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, x, weight, dout , bias):
        bias = None
        x_t = x
        weight_t = weight
        dout_t = dout
        bias_t = bias
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            weight_t = paddle.cast(weight, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
            if bias is not None:
                bias_t = paddle.cast(bias, dtype="uint16")
            else:
                bias_t = bias
        out = paddle.nn.functional.linear(x_t, weight_t, bias_t)
        if bias is not None:
            out_grads = paddle.grad(
                [out], [x, weight , bias], grad_outputs=[dout_t]
            )
        else:
            out_grads = paddle.grad(
                [out], [x, weight], grad_outputs=[dout_t]
            )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, weight_eager, dout_eager , bias_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, weight_eager, dout_eager , bias_eager)
        del x_eager
        del weight_eager
        del dout_eager
        del bias_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager,
                            )
        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()
        # save eager res for test_linear_incubate

        # compare eager res with torch
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
            api="paddle.nn.functional.linear",
        )
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
                api="paddle.nn.functional.linear",
            )

test_shapes=[[],[[256,10,4096],[4096,12288],[12288]],
[[10,181,1536],[1536,4608],None],
[[256,10,16384],[16384,4096],[4096]],
[[10,181,1536],[1536,1536],[1536]],
[[10,181,3072],[3072,1536],[1536]],
[[10,181,1536],[1536,4096],None],
[[10,181,4096],[4096,1536],None],
[[10,181,1536],[1536,1536],None],
[[256,10,4096],[4096,16384],[16384]],
[[10,181,1536],[1536,64],[64]],
[[256,10,4096],[4096,4096],[4096]],
[[10,256,4096],[4096,1536],None]]

class TestLinearDevelopCase1_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[1][0]
        self.shape_w = test_shapes[1][1]
        self.shape_b = test_shapes[1][2]

class TestLinearDevelopCase1_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[1][0]
        self.shape_w = test_shapes[1][1]
        self.shape_b = test_shapes[1][2]

class TestLinearDevelopCase1_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[1][0]
        self.shape_w = test_shapes[1][1]
        self.shape_b = test_shapes[1][2]

class TestLinearDevelopCase2_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[2][0]
        self.shape_w = test_shapes[2][1]
        self.shape_b = test_shapes[2][2]

class TestLinearDevelopCase2_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[2][0]
        self.shape_w = test_shapes[2][1]
        self.shape_b = test_shapes[2][2]

class TestLinearDevelopCase2_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[2][0]
        self.shape_w = test_shapes[2][1]
        self.shape_b = test_shapes[2][2]

class TestLinearDevelopCase3_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[3][0]
        self.shape_w = test_shapes[3][1]
        self.shape_b = test_shapes[3][2]

class TestLinearDevelopCase3_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[3][0]
        self.shape_w = test_shapes[3][1]
        self.shape_b = test_shapes[3][2]

class TestLinearDevelopCase3_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[3][0]
        self.shape_w = test_shapes[3][1]
        self.shape_b = test_shapes[3][2]

class TestLinearDevelopCase4_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[4][0]
        self.shape_w = test_shapes[4][1]
        self.shape_b = test_shapes[4][2]

class TestLinearDevelopCase4_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[4][0]
        self.shape_w = test_shapes[4][1]
        self.shape_b = test_shapes[4][2]

class TestLinearDevelopCase4_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[4][0]
        self.shape_w = test_shapes[4][1]
        self.shape_b = test_shapes[4][2]

class TestLinearDevelopCase5_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[5][0]
        self.shape_w = test_shapes[5][1]
        self.shape_b = test_shapes[5][2]

class TestLinearDevelopCase5_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[5][0]
        self.shape_w = test_shapes[5][1]
        self.shape_b = test_shapes[5][2]

class TestLinearDevelopCase5_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[5][0]
        self.shape_w = test_shapes[5][1]
        self.shape_b = test_shapes[5][2]
class TestLinearDevelopCase6_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[6][0]
        self.shape_w = test_shapes[6][1]
        self.shape_b = test_shapes[6][2]

class TestLinearDevelopCase6_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[6][0]
        self.shape_w = test_shapes[6][1]
        self.shape_b = test_shapes[6][2]

class TestLinearDevelopCase6_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[6][0]
        self.shape_w = test_shapes[6][1]
        self.shape_b = test_shapes[6][2]
class TestLinearDevelopCase7_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[7][0]
        self.shape_w = test_shapes[7][1]
        self.shape_b = test_shapes[7][2]

class TestLinearDevelopCase7_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[7][0]
        self.shape_w = test_shapes[7][1]
        self.shape_b = test_shapes[7][2]

class TestLinearDevelopCase7_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[7][0]
        self.shape_w = test_shapes[7][1]
        self.shape_b = test_shapes[7][2]
class TestLinearDevelopCase8_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[8][0]
        self.shape_w = test_shapes[8][1]
        self.shape_b = test_shapes[8][2]

class TestLinearDevelopCase8_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[8][0]
        self.shape_w = test_shapes[8][1]
        self.shape_b = test_shapes[8][2]

class TestLinearDevelopCase8_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[8][0]
        self.shape_w = test_shapes[8][1]
        self.shape_b = test_shapes[8][2]
class TestLinearDevelopCase9_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[9][0]
        self.shape_w = test_shapes[9][1]
        self.shape_b = test_shapes[9][2]

class TestLinearDevelopCase9_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[9][0]
        self.shape_w = test_shapes[9][1]
        self.shape_b = test_shapes[9][2]

class TestLinearDevelopCase9_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[9][0]
        self.shape_w = test_shapes[9][1]
        self.shape_b = test_shapes[9][2]
class TestLinearDevelopCase10_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[10][0]
        self.shape_w = test_shapes[10][1]
        self.shape_b = test_shapes[10][2]

class TestLinearDevelopCase10_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[10][0]
        self.shape_w = test_shapes[10][1]
        self.shape_b = test_shapes[10][2]

class TestLinearDevelopCase10_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[10][0]
        self.shape_w = test_shapes[10][1]
        self.shape_b = test_shapes[10][2]
class TestLinearDevelopCase11_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[11][0]
        self.shape_w = test_shapes[11][1]
        self.shape_b = test_shapes[11][2]

class TestLinearDevelopCase11_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[11][0]
        self.shape_w = test_shapes[11][1]
        self.shape_b = test_shapes[11][2]

class TestLinearDevelopCase11_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[11][0]
        self.shape_w = test_shapes[11][1]
        self.shape_b = test_shapes[11][2]
class TestLinearDevelopCase12_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.shape_x = test_shapes[12][0]
        self.shape_w = test_shapes[12][1]
        self.shape_b = test_shapes[12][2]

class TestLinearDevelopCase12_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.shape_x = test_shapes[12][0]
        self.shape_w = test_shapes[12][1]
        self.shape_b = test_shapes[12][2]

class TestLinearDevelopCase12_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.shape_x = test_shapes[12][0]
        self.shape_w = test_shapes[12][1]
        self.shape_b = test_shapes[12][2]

if __name__ == '__main__':
    unittest.main()
