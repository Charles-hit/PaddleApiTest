import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type


def generate_np_inputs_and_dout():
    x_case1 = np.random.random(size=[1, 4096, 4096]).astype("float32") - 0.5
    scale_case1 = 10000.0
    bias_case1 = -1.0
    dout_case1 = np.random.random(size=[1, 4096, 4096]).astype("float32") - 0.5

    x_case2 = np.random.random(size=[1, 32, 4096, 192]).astype("float32") - 0.5
    scale_case2 = 0.07216878364870322
    bias_case2 = 0.0
    dout_case2 = np.random.random(size=[1, 32, 4096, 192]).astype("float32") - 0.5

    x_case3 = np.random.random(size=[14336, 31250]).astype("float32") - 0.5
    scale_case3 = 0.004821738581303281 
    bias_case3 = 0.0
    dout_case3 = np.random.random(size=[14336, 31250]).astype("float32") - 0.5

    x_case4 = np.random.random(size=[14336, 5376]).astype("float32") - 0.5
    scale_case4 = 0.004821738581303281 
    bias_case4 = 0.0
    dout_case4 = np.random.random(size=[14336, 5376]).astype("float32") - 0.5

    x_case5 = np.random.random(size=[14336, 9632]).astype("float32") - 0.5
    scale_case5 = 0.004821738581303281
    bias_case5 = 0.0
    dout_case5 = np.random.random(size=[14336, 9632]).astype("float32") - 0.5

    x_case6 = np.random.random(size=[1792, 14336]).astype("float32") - 0.5
    scale_case6 = 0.004821738581303281 
    bias_case6 = 0.0
    dout_case6 = np.random.random(size=[1792, 14336]).astype("float32") - 0.5

    x_case7 = np.random.random(size=[31250, 14336]).astype("float32") - 0.5
    scale_case7 = 0.004821738581303281 
    bias_case7 = 0.0
    dout_case7 = np.random.random(size=[31250, 14336]).astype("float32") - 0.5

    x_case8 = np.random.random(size=[4816, 14336]).astype("float32") - 0.5
    scale_case8 = 0.004821738581303281
    bias_case8 = 0.0
    dout_case8 = np.random.random(size=[4816, 14336]).astype("float32") - 0.5

    x_case9 = np.random.random(size=[14336]).astype("float32") - 0.5
    scale_case9 = 0.125
    bias_case9 = 0.0
    dout_case9 = np.random.random(size=[14336]).astype("float32") - 0.5

    np.savez("./inputs_case1.npz", x=x_case1, scale=scale_case1, bias=bias_case1, dout = dout_case1)
    np.savez("./inputs_case2.npz", x=x_case2, scale=scale_case2, bias=bias_case2, dout = dout_case2)
    np.savez("./inputs_case3.npz", x=x_case3, scale=scale_case3, bias=bias_case3, dout = dout_case3)
    np.savez("./inputs_case4.npz", x=x_case4, scale=scale_case4, bias=bias_case4, dout = dout_case4)
    np.savez("./inputs_case5.npz", x=x_case5, scale=scale_case5, bias=bias_case5, dout = dout_case5)
    np.savez("./inputs_case6.npz", x=x_case6, scale=scale_case6, bias=bias_case6, dout = dout_case6)
    np.savez("./inputs_case7.npz", x=x_case7, scale=scale_case7, bias=bias_case7, dout = dout_case7)
    np.savez("./inputs_case8.npz", x=x_case8, scale=scale_case8, bias=bias_case8, dout = dout_case8)
    np.savez("./inputs_case9.npz", x=x_case9, scale=scale_case9, bias=bias_case9, dout = dout_case9)


class TestScaleDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, scale_torch, bias_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(x_torch, scale_torch, bias_torch, self.bias_after_scale, dout_torch)
        del x_torch 
        del scale_torch
        del bias_torch
        del dout_torch 
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
                                lambda x: x.cpu().numpy(),
                                out_grads_torch,
                            )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float32"
        self.bias_after_scale = False
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp32.npz"
    
    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_x = np_inputs_array["x"]
        self.np_scale = float(np_inputs_array["scale"])
        self.np_bias = float(np_inputs_array["bias"])
        self.np_dout = np_inputs_array["dout"]
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
        scale_torch = self.np_scale
        bias_torch = self.np_bias
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, scale_torch, bias_torch, dout_torch
    
    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        scale_eager = self.np_scale
        bias_eager = self.np_bias
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, scale_eager, bias_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        scale_static = self.np_scale
        bias_static = self.np_bias
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, scale_static, bias_static, dout_static

    def cal_torch_res(self, x, scale, bias, bias_after_scale, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        scale = torch.tensor(scale, dtype=convert_dtype_to_torch_type(self.dtype))
        bias = torch.tensor(bias, dtype=convert_dtype_to_torch_type(self.dtype))
        if bias_after_scale:
            out = x * scale + bias
        else:
            out = scale * (x + bias)
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, scale, bias, bias_after_scale, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.scale(x, scale, bias, bias_after_scale)
        out_grads = paddle.grad(
            [out], [x], grad_outputs=[dout]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def cal_static_res(self, x, scale, bias, bias_after_scale, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.scale(x, scale, bias, bias_after_scale)
        out_grads = paddle.static.gradients(
            [out], [x], target_gradients=[dout]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, scale_eager, bias_eager, dout_eager = self.gen_eager_inputs_and_dout()

        out_eager, out_grads_eager = self.cal_eager_res(x_eager, scale_eager, bias_eager, self.bias_after_scale, dout_eager)
        del x_eager
        del scale_eager
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
        # save eager res for test_scale_incubate
        np.savez(self.save_eager_res_path, out_eager=out_eager_np, out_grads_eager_0=out_grads_eager_np[0])
        
        max_atol_idx = np.argmax(np.abs(out_eager_np-self.out_torch))
        max_rtol_idx = np.argmax(np.abs((out_eager_np-self.out_torch)/out_eager_np))
        # compare eager res with torch
        np.testing.assert_allclose(
            out_eager_np,
            self.out_torch,
            self.atol,
            self.rtol,
            err_msg=(
                'Develop: compare scale eager forward res with torch failed in %s dtype,\n'
                'max_atol_idx: %d, eager_value: %d, torch_value: %d, \n'
                'max_rtol_idx: %d, eager_value: %d, torch_value: %d, \n'
            )
            % (self.dtype, max_atol_idx, out_eager_np.flatten()[max_atol_idx].item(), self.out_torch.flatten()[max_atol_idx].item(),
                max_rtol_idx, out_eager_np.flatten()[max_rtol_idx].item(), self.out_torch.flatten()[max_rtol_idx].item()),
        )
        for idx in range(len(out_grads_eager_np)):
            max_atol_idx = np.argmax(np.abs(out_grads_eager_np[idx]-self.out_grads_torch[idx]))
            max_rtol_idx = np.argmax(np.abs((out_grads_eager_np[idx]-self.out_grads_torch[idx])/out_grads_eager_np[idx]))
            np.testing.assert_allclose(
                out_grads_eager_np[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                err_msg=(
                    'Develop: compare scale eager grad res with torch failed in %s dtype,\n'
                    'max_atol_idx: %d, eager_value: %d, torch_value: %d, \n'
                    'max_rtol_idx: %d, eager_value: %d, torch_value: %d, \n'
                )
            % (self.dtype, max_atol_idx, out_grads_eager_np[idx].flatten()[max_atol_idx].item(), self.out_grads_torch[idx].flatten()[max_atol_idx].item(),
                max_rtol_idx, out_grads_eager_np[idx].flatten()[max_rtol_idx].item(), self.out_grads_torch[idx].flatten()[max_rtol_idx].item()),
            )
    
    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, scale_static, bias_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    scale_static,
                    bias_static,
                    self.bias_after_scale,
                    dout_static,
            )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # save static res for test_scale_incubate
        np.savez(self.save_static_res_path, out_static=out_static, out_grads_static_0=out_grads_static[0])
        
        max_atol_idx = np.argmax(np.abs(out_static-self.out_torch))
        max_rtol_idx = np.argmax(np.abs((out_static-self.out_torch)/out_static))
        # compare static res with torch
        np.testing.assert_allclose(
            out_static,
            self.out_torch,
            self.atol,
            self.rtol,
            err_msg=(
                'Develop: compare scale static forward res with torch failed in %s dtype\n'
                'max_atol_idx: %d, static_value: %d, torch_value: %d, \n'
                'max_rtol_idx: %d, static_value: %d, torch_value: %d, \n'
            )
            % (self.dtype, max_atol_idx, out_static.flatten()[max_atol_idx].item(), self.out_torch.flatten()[max_atol_idx].item(),
                max_rtol_idx, out_static.flatten()[max_rtol_idx].item(), self.out_torch.flatten()[max_rtol_idx].item()),
        )
        for idx in range(len(out_grads_static)):
            max_atol_idx = np.argmax(np.abs(out_grads_static[idx]-self.out_grads_torch[idx]))
            max_rtol_idx = np.argmax(np.abs((out_grads_static[idx]-self.out_grads_torch[idx])/out_grads_static[idx]))
            np.testing.assert_allclose(
                out_grads_static[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                err_msg=(
                    'Develop: compare scale static grad res with torch failed in %s dtype\n'
                    'max_atol_idx: %d, static_value: %d, torch_value: %d, \n'
                    'max_rtol_idx: %d, static_value: %d, torch_value: %d, \n'
                )
            % (self.dtype, max_atol_idx, out_grads_static[idx].flatten()[max_atol_idx].item(), self.out_grads_torch[idx].flatten()[max_atol_idx].item(),
                max_rtol_idx, out_grads_static[idx].flatten()[max_rtol_idx].item(), self.out_grads_torch[idx].flatten()[max_rtol_idx].item()),
            )

    def test_eager_stability(self):
        x_eager, scale_eager, bias_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(x_eager, scale_eager, bias_eager, self.bias_after_scale, dout_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager_baseline,
                            )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager, out_grads_eager = self.cal_eager_res(x_eager, scale_eager, bias_eager, self.bias_after_scale, dout_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                                    lambda x: x.numpy(),
                                    out_grads_eager,
                                )
            max_atol_idx = np.argmax(np.abs(out_eager-out_eager_baseline_np))
            max_rtol_idx = np.argmax(np.abs((out_eager-out_eager_baseline_np)/out_eager))
            np.testing.assert_equal(
                out_eager,
                out_eager_baseline_np,
                err_msg=(
                    'Develop: paddle.scale eager forward is unstable in %s dtype\n'
                    'max_atol_idx: %d, eager_value: %d, eager_baseline_value: %d, \n'
                    'max_rtol_idx: %d, eager_value: %d, eager_baseline_value: %d, \n'
                )
                % (self.dtype, max_atol_idx, out_eager.flatten()[max_atol_idx].item(), out_eager_baseline_np.flatten()[max_atol_idx].item(),
                    max_rtol_idx, out_eager.flatten()[max_rtol_idx].item(), out_eager_baseline_np.flatten()[max_rtol_idx].item()),
            )
            for idx in range(len(out_grads_eager)):
                max_atol_idx = np.argmax(np.abs(out_grads_eager[idx]-out_grads_eager_baseline_np[idx]))
                max_rtol_idx = np.argmax(np.abs((out_grads_eager[idx]-out_grads_eager_baseline_np[idx])/out_grads_eager[idx]))
                np.testing.assert_equal(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    err_msg=(
                        'Develop: paddle.scale eager grad is unstable in %s dtype\n'
                        'max_atol_idx: %d, eager_value: %d, eager_baseline_value: %d, \n'
                        'max_rtol_idx: %d, eager_value: %d, eager_baseline_value: %d, \n'
                    )
                % (self.dtype, max_atol_idx, out_grads_eager[idx].flatten()[max_atol_idx].item(), out_grads_eager_baseline_np[idx].flatten()[max_atol_idx].item(),
                    max_rtol_idx, out_grads_eager[idx].flatten()[max_rtol_idx].item(), out_grads_eager_baseline_np[idx].flatten()[max_rtol_idx].item()),
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, scale_static, bias_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    scale_static,
                    bias_static,
                    self.bias_after_scale,
                    dout_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(5):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                max_atol_idx = np.argmax(np.abs(out_static-out_static_baseline))
                max_rtol_idx = np.argmax(np.abs((out_static-out_static_baseline)/out_static))
                np.testing.assert_equal(
                    out_static,
                    out_static_baseline,
                    err_msg=(
                        'Develop: paddle.scale static forward is unstable in %s dtype\n'
                        'max_atol_idx: %d, static_value: %d, eager_baseline_value: %d, \n'
                        'max_rtol_idx: %d, static_value: %d, eager_baseline_value: %d, \n'
                    )
                    % (self.dtype, max_atol_idx, out_static.flatten()[max_atol_idx].item(), out_static_baseline.flatten()[max_atol_idx].item(),
                        max_rtol_idx, out_static.flatten()[max_rtol_idx].item(), out_static_baseline.flatten()[max_rtol_idx].item()),
                )
                for idx in range(len(out_grads_static)):
                    max_atol_idx = np.argmax(np.abs(out_grads_static[idx]-out_grads_static_baseline[idx]))
                    max_rtol_idx = np.argmax(np.abs((out_grads_static[idx]-out_grads_static_baseline[idx])/out_grads_static[idx]))
                    np.testing.assert_equal(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        err_msg=(
                            'Develop: paddle.scale static grad is unstable in %s dtype\n'
                            'max_atol_idx: %d, static_value: %d, static_baseline_value: %d, \n'
                            'max_rtol_idx: %d, static_value: %d, static_baseline_value: %d, \n'
                        )
                    % (self.dtype, max_atol_idx, out_grads_static[idx].flatten()[max_atol_idx].item(), out_grads_static_baseline[idx].flatten()[max_atol_idx].item(),
                        max_rtol_idx, out_grads_static[idx].flatten()[max_rtol_idx].item(), out_grads_static_baseline[idx].flatten()[max_rtol_idx].item()),
                    )


class TestScaleDevelopCase1_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.bias_after_scale = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"


class TestScaleDevelopCase1_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.bias_after_scale = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"


class TestScaleDevelopCase2_FP32(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.bias_after_scale = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp32.npz"


class TestScaleDevelopCase2_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp16.npz"


class TestScaleDevelopCase2_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npz"


class TestScaleDevelopCase3_FP32(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.bias_after_scale = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case3_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_fp32.npz"


class TestScaleDevelopCase3_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case3_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_fp16.npz"


class TestScaleDevelopCase3_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case3_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_bfp16.npz"


class TestScaleDevelopCase4_FP32(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case4.npz"
        self.bias_after_scale = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case4_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case4_fp32.npz"


class TestScaleDevelopCase4_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case4.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case4_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case4_fp16.npz"


class TestScaleDevelopCase4_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case4.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case4_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case4_bfp16.npz"


class TestScaleDevelopCase5_FP32(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case5.npz"
        self.bias_after_scale = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case5_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case5_fp32.npz"


class TestScaleDevelopCase5_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case5.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case5_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case5_fp16.npz"


class TestScaleDevelopCase5_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case5.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case5_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case5_bfp16.npz"


class TestScaleDevelopCase6_FP32(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case6.npz"
        self.bias_after_scale = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case6_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case6_fp32.npz"


class TestScaleDevelopCase6_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case6.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case6_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case6_fp16.npz"


class TestScaleDevelopCase6_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case6.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case6_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case6_bfp16.npz"


class TestScaleDevelopCase7_FP32(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case7.npz"
        self.bias_after_scale = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case7_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case7_fp32.npz"


class TestScaleDevelopCase7_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case7.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case7_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case7_fp16.npz"


class TestScaleDevelopCase7_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case7.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case7_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case7_bfp16.npz"


class TestScaleDevelopCase8_FP32(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case8.npz"
        self.bias_after_scale = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case8_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case8_fp32.npz"


class TestScaleDevelopCase8_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case8.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case8_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case8_fp16.npz"


class TestScaleDevelopCase8_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case8.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case8_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case8_bfp16.npz"


class TestScaleDevelopCase9_FP32(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case9.npz"
        self.bias_after_scale = True
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case9_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case9_fp32.npz"


class TestScaleDevelopCase9_FP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case9.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case9_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case9_fp16.npz"


class TestScaleDevelopCase9_BFP16(TestScaleDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case9.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case9_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case9_bfp16.npz"


if __name__ == '__main__':
    generate_np_inputs_and_dout()
    unittest.main()
