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

class TestConvndDevelopCase1_BFP16(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, weight_torch, bias_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, weight_torch, bias_torch, dout_torch
        )
        del x_torch
        del weight_torch
        del bias_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().detach().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "bfloat16"
        self.x_shape = [1, 3, 320, 576]
        self.weight_shape = [128, 3, 3, 3]
        self.bias_shape = [128]
        self.stride = [1, 1]
        self.padding = [1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 128, 320, 576]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]


    def init_np_inputs_and_dout(self):
        self.np_x = np.random.random(size=self.x_shape).astype("float32") - 0.5
        self.np_weight = np.random.random(size=self.weight_shape).astype("float32") - 0.5
        self.np_bias = np.random.random(size=self.bias_shape).astype("float32") - 0.5
        self.np_dout = np.random.random(size=self.out_shape).astype("float32") - 0.5
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_weight = self.np_weight.astype("float16")
            self.np_bias = self.np_bias.astype("float16")
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
        weight_torch = torch.tensor(
            self.np_weight,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        bias_torch = torch.tensor(
            self.np_bias,
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
        return x_torch, weight_torch, bias_torch, dout_torch

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
        bias_eager = paddle.to_tensor(
            self.np_bias,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        bias_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, weight_eager, bias_eager, dout_eager

    def cal_torch_res(self, x, weight, bias, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            weight = weight.to(dtype=torch.bfloat16)
            bias = bias.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        if self.op_type == 'conv2d':
            torch_api = torch.nn.functional.conv2d
        elif self.op_type == 'conv3d':
            torch_api = torch.nn.functional.conv3d
        out = torch_api(input=x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        out_grads = torch.autograd.grad([out], [x, weight, bias], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, weight, bias, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            weight = paddle.cast(weight, dtype="uint16")
            bias = paddle.cast(bias, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.nn.functional.conv._conv_nd(
            x=x, 
            weight=weight, 
            bias=bias, 
            stride=self.stride, 
            padding=self.padding, 
            padding_algorithm=self.padding_algorithm, 
            dilation=self.dilation, 
            groups=self.groups, 
            data_format=self.data_format, 
            channel_dim=self.channel_dim,
            op_type=self.op_type, 
            use_cudnn=self.use_cudnn
        )
        out_grads = paddle.grad([out], [x, weight, bias], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, weight_eager, bias_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, weight_eager, bias_eager, dout_eager
        )
        del x_eager
        del weight_eager
        del bias_eager
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
            api="paddle.nn.functional.conv._conv_nd",
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
                api="paddle.nn.functional.conv._conv_nd",
            )


# error
class TestConvndDevelopCase1_FP32(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float32"
        self.x_shape = [1, 3, 320, 576]
        self.weight_shape = [128, 3, 3, 3]
        self.bias_shape = [128]
        self.stride = [1, 1]
        self.padding = [1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 128, 320, 576]

class TestConvndDevelopCase1_FP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float16"
        self.x_shape = [1, 3, 320, 576]
        self.weight_shape = [128, 3, 3, 3]
        self.bias_shape = [128]
        self.stride = [1, 1]
        self.padding = [1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 128, 320, 576]


class TestConvndDevelopCase2_BFP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.x_shape = [1, 8, 40, 72]
        self.weight_shape = [8, 8, 1, 1]
        self.bias_shape = [8]
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 8, 40, 72]

# error
class TestConvndDevelopCase2_FP32(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float32"
        self.x_shape = [1, 8, 40, 72]
        self.weight_shape = [8, 8, 1, 1]
        self.bias_shape = [8]
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 8, 40, 72]

class TestConvndDevelopCase2_FP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float16"
        self.x_shape = [1, 8, 40, 72]
        self.weight_shape = [8, 8, 1, 1]
        self.bias_shape = [8]
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 8, 40, 72]





class TestConvndDevelopCase3_BFP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.x_shape = [10, 4, 1, 40, 72]
        self.weight_shape = [1536, 4, 1, 4, 4]
        self.bias_shape = [1536]
        self.stride = [1, 4, 4]
        self.padding = [0, 0, 0]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1, 1]
        self.groups = 1
        self.data_format = 'NCDHW'
        self.channel_dim = 1
        self.op_type = 'conv3d'
        self.use_cudnn = True
        self.out_shape = [10, 1536, 1, 10, 18]

# error
class TestConvndDevelopCase3_FP32(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float32"
        self.x_shape = [10, 4, 1, 40, 72]
        self.weight_shape = [1536, 4, 1, 4, 4]
        self.bias_shape = [1536]
        self.stride = [1, 4, 4]
        self.padding = [0, 0, 0]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1, 1]
        self.groups = 1
        self.data_format = 'NCDHW'
        self.channel_dim = 1
        self.op_type = 'conv3d'
        self.use_cudnn = True
        self.out_shape = [10, 1536, 1, 10, 18]

class TestConvndDevelopCase3_FP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float16"
        self.x_shape = [10, 4, 1, 40, 72]
        self.weight_shape = [1536, 4, 1, 4, 4]
        self.bias_shape = [1536]
        self.stride = [1, 4, 4]
        self.padding = [0, 0, 0]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1, 1]
        self.groups = 1
        self.data_format = 'NCDHW'
        self.channel_dim = 1
        self.op_type = 'conv3d'
        self.use_cudnn = True
        self.out_shape = [10, 1536, 1, 10, 18]


class TestConvndDevelopCase4_BFP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.x_shape = [10, 4, 1, 40, 72]
        self.weight_shape = [4, 4, 3, 3, 3]
        self.bias_shape = [4]
        self.stride = [1, 1, 1]
        self.padding = [1, 1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1, 1]
        self.groups = 1
        self.data_format = 'NCDHW'
        self.channel_dim = 1
        self.op_type = 'conv3d'
        self.use_cudnn = True
        self.out_shape = [10, 4, 1, 40, 72]


# error
class TestConvndDevelopCase4_FP32(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float32"
        self.x_shape = [10, 4, 1, 40, 72]
        self.weight_shape = [4, 4, 3, 3, 3]
        self.bias_shape = [4]
        self.stride = [1, 1, 1]
        self.padding = [1, 1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1, 1]
        self.groups = 1
        self.data_format = 'NCDHW'
        self.channel_dim = 1
        self.op_type = 'conv3d'
        self.use_cudnn = True
        self.out_shape = [10, 4, 1, 40, 72]

class TestConvndDevelopCase4_FP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float16"
        self.x_shape = [10, 4, 1, 40, 72]
        self.weight_shape = [4, 4, 3, 3, 3]
        self.bias_shape = [4]
        self.stride = [1, 1, 1]
        self.padding = [1, 1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1, 1]
        self.groups = 1
        self.data_format = 'NCDHW'
        self.channel_dim = 1
        self.op_type = 'conv3d'
        self.use_cudnn = True
        self.out_shape = [10, 4, 1, 40, 72]


class TestConvndDevelopCase5_BFP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "bfloat16"
        self.x_shape = [1, 512, 40, 72]
        self.weight_shape = [8, 512, 3, 3]
        self.bias_shape = [8]
        self.stride = [1, 1]
        self.padding = [1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 8, 40, 72]

class TestConvndDevelopCase5_FP32(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float32"
        self.x_shape = [1, 512, 40, 72]
        self.weight_shape = [8, 512, 3, 3]
        self.bias_shape = [8]
        self.stride = [1, 1]
        self.padding = [1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 8, 40, 72]

class TestConvndDevelopCase5_FP16(TestConvndDevelopCase1_BFP16):
    def init_params(self):
        self.dtype = "float16"
        self.x_shape = [1, 512, 40, 72]
        self.weight_shape = [8, 512, 3, 3]
        self.bias_shape = [8]
        self.stride = [1, 1]
        self.padding = [1, 1]
        self.padding_algorithm = 'EXPLICIT'
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = 'NCHW'
        self.channel_dim = 1
        self.op_type = 'conv2d'
        self.use_cudnn = True
        self.out_shape = [1, 8, 40, 72]


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()