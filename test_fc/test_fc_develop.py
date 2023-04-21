
import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type


def generate_np_inputs_and_dout():
    x_case1 = np.random.random(size=[1, 4096, 12288]).astype("float32")
    w_case1 = np.random.random(size=[12288, 12288]).astype("float32")
    b_case1 = np.random.random(size=[12288]).astype("float32")
    dout_case1 = np.random.random(size=[1, 4096, 12288]).astype("float32")

    x_case2 = np.random.random(size=[1, 4096, 12288]).astype("float32")
    w_case2 = np.random.random(size=[12288, 49152]).astype("float32")
    b_case2 = np.random.random(size=[49152]).astype("float32")
    dout_case2 = np.random.random(size=[1, 4096, 49152]).astype("float32")

    x_case3 = np.random.random(size=[1, 4096, 49152]).astype("float32")
    w_case3 = np.random.random(size=[49152, 12288]).astype("float32")
    b_case3 = np.random.random(size=[12288]).astype("float32")
    dout_case3 = np.random.random(size=[1, 4096, 12288]).astype("float32")

    np.savez("./inputs_case1.npz", x = x_case1, w = w_case1, b = b_case1, dout = dout_case1)
    np.savez("./inputs_case2.npz", x = x_case2, w = w_case2, b = b_case2, dout = dout_case2)
    np.savez("./inputs_case3.npz", x = x_case3, w = w_case3, b = b_case3, dout = dout_case3)

class TestFCDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, w_torch, b_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(x_torch, w_torch, b_torch, dout_torch)
        del x_torch 
        del w_torch 
        del b_torch
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
        self.num_flatten_dims = -1
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp32.npz"
    
    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_x = np_inputs_array["x"]
        self.np_w = np_inputs_array["w"]
        self.np_b = np_inputs_array["b"]
        self.np_dout = np_inputs_array["dout"]
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_w = self.np_w.astype("float16")
            self.np_b = self.np_b.astype("float16")
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
        w_torch = torch.tensor(
            self.np_w,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        b_torch = torch.tensor(
            self.np_b,
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
        return x_torch, w_torch, b_torch, dout_torch
    
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

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        w_static = paddle.static.data(
            'w',
            shape=self.np_w.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        w_static.stop_gradient = False
        b_static = paddle.static.data(
            'b',
            shape=self.np_b.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        b_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, w_static, b_static, dout_static

    def cal_torch_res(self, x, w, b, dout):
        x_t = x
        w_t = w
        b_t = b
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            w_t = w.to(dtype=torch.bfloat16)
            b_t = b.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        
        w_t = torch.transpose (w_t, 0, 1)
        out = torch.nn.functional.linear(x_t, w_t, b_t)
        out_grads = torch.autograd.grad([out], [x, w, b], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, x, w, b, dout):
        x_t = x
        w_t = w
        b_t = b
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            w_t = paddle.cast(w, dtype="uint16")
            b_t = paddle.cast(b, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.incubate.nn.functional.fused_linear(x_t, w_t, b_t)
        out_grads = paddle.grad(
            [out], [x, w, b], grad_outputs=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def cal_static_res(self, x, w, b, dout):
        x_t = x
        w_t = w
        b_t = b
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            w_t = paddle.cast(w, dtype="uint16")
            b_t = paddle.cast(b, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.incubate.nn.functional.fused_linear(x_t, w_t, b_t)
        out_grads = paddle.static.gradients(
            [out], [x, w, b], target_gradients=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
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
        out_grads_eager_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager,
                            )
        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()
        # save eager res for test_matmul_incubate
        np.savez(self.save_eager_res_path, 
                out_eager=out_eager_np, 
                out_grads_eager_0=out_grads_eager_np[0], 
                out_grads_eager_1=out_grads_eager_np[1], 
                out_grads_eager_2=out_grads_eager_np[2])
        
        # compare eager res with torch
        np.testing.assert_allclose(
            out_eager_np,
            self.out_torch,
            self.atol,
            self.rtol,
            err_msg=(
                'Develop: compare paddle.incubate.nn.functional.fused_linear eager forward res with torch failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_eager_np)):
            np.testing.assert_allclose(
                out_grads_eager_np[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                err_msg=(
                    'Develop: compare paddle.incubate.nn.functional.fused_linear eager grad res with torch failed in %s dtype'
                )
                % self.dtype,
            )
    
    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, w_static, b_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    w_static,
                    b_static,                   
                    dout_static,
            )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "w": self.np_w, "b": self.np_b, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # save static res for test_matmul_incubate
        np.savez(self.save_static_res_path, 
                out_static=out_static, 
                out_grads_static_0=out_grads_static[0], 
                out_grads_static_1=out_grads_static[1],
                out_grads_static_2=out_grads_static[2])
        
        # compare static res with torch
        np.testing.assert_allclose(
            out_static,
            self.out_torch,
            self.atol,
            self.rtol,
            err_msg=(
                'Develop: compare paddle.incubate.nn.functional.fused_linear static forward res with torch failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_static)):
            np.testing.assert_allclose(
                out_grads_static[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                err_msg=(
                    'Develop: compare paddle.incubate.nn.functional.fused_linear static grad res with torch failed in %s dtype'
                )
                % self.dtype,
            )

    def test_eager_stability(self):
        x_eager, w_eager, b_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(x_eager, w_eager, b_eager, dout_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager_baseline,
                            )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(x_eager, w_eager, b_eager, dout_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                                    lambda x: x.numpy(),
                                    out_grads_eager,
                                )
            np.testing.assert_equal(
                out_eager,
                out_eager_baseline_np,
                err_msg=(
                    'Develop: paddle.incubate.nn.functional.fused_linear eager forward is unstable in %s dtype'
                )
                % self.dtype,
            )
            for idx in range(len(out_grads_eager)):
                np.testing.assert_equal(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    err_msg=(
                        'Develop: paddle.incubate.nn.functional.fused_linear eager grad is unstable in %s dtype'
                    )
                    % self.dtype,
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, w_static, b_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    w_static,
                    b_static,
                    dout_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "w": self.np_w, "b": self.np_b, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "w": self.np_w, "b": self.np_b, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                np.testing.assert_equal(
                    out_static,
                    out_static_baseline,
                    err_msg=(
                        'Develop: paddle.incubate.nn.functional.fused_linear static forward is unstable in %s dtype'
                    )
                    % self.dtype,
                )
                for idx in range(len(out_grads_static)):
                    np.testing.assert_equal(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        err_msg=(
                            'Develop: paddle.incubate.nn.functional.fused_linear static grad is unstable in %s dtype'
                        )
                        % self.dtype,
                    )

class TestFCDevelopCase1_FP16(TestFCDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.num_flatten_dims = -1
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"

class TestFCDevelopCase1_BFP16(TestFCDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.num_flatten_dims = -1
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"

class TestFCDevelopCase2_FP32(TestFCDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.num_flatten_dims = -1
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp32.npz"

class TestFCDevelopCase2_FP16(TestFCDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.num_flatten_dims = -1
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp16.npz"

class TestFCDevelopCase2_BFP16(TestFCDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.num_flatten_dims = -1
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npz"

class TestFCDevelopCase3_FP32(TestFCDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.num_flatten_dims = -1
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case3_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_fp32.npz"

class TestFCDevelopCase3_FP16(TestFCDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.num_flatten_dims = -1
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case3_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_fp16.npz"

class TestFCDevelopCase3_BFP16(TestFCDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.num_flatten_dims = -1
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case3_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_bfp16.npz"

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    generate_np_inputs_and_dout()
    unittest.main()
