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

class TestSquaredl2NormDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, dout_torch
        )
        del x_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[12288, 1792]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
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
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, dout_static

    def cal_torch_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        out = torch.square(x).sum().reshape(1)
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.nn.clip._squared_l2_norm(x)
        out_grads = paddle.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def cal_static_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.incubate.nn.functional.silu(x)
        out_grads = paddle.static.gradients(
            [out], [x], target_gradients=[dout]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, dout_eager
        )
        del x_eager
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
            api="paddle.nn.functional.silu",
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
                api="paddle.nn.functional.silu",
            )

    # def test_static_accuracy(self):
    #     with paddle.fluid.framework._dygraph_guard(None):
    #         mp, sp = paddle.static.Program(), paddle.static.Program()
    #         with paddle.static.program_guard(mp, sp):
    #             (
    #                 x_static,
    #                 dout_static,
    #             ) = self.gen_static_inputs_and_dout()
    #             (out_static, out_grads_static) = self.cal_static_res(
    #                 x_static,
    #                 dout_static,
    #             )
    #         exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
    #         exe.run(sp)
    #         out = exe.run(
    #             mp,
    #             feed={"x": self.np_x, "dout": self.np_dout},
    #             fetch_list=[out_static] + out_grads_static,
    #         )
    #         out_static, out_grads_static = out[0], out[1:]

    #     # compare develop static forward res with torch
    #     np_assert_accuracy(
    #         out_static,
    #         self.out_torch,
    #         self.atol,
    #         self.rtol,
    #         self.dtype,
    #         version_a="paddle_develop",
    #         version_b="torch",
    #         eager_or_static_mode="static",
    #         fwd_or_bkd="forward",
    #         api="paddle.nn.functional.silu",
    #     )
    #     # compare develop static backward res with torch
    #     for idx in range(len(out_grads_static)):
    #         np_assert_accuracy(
    #             out_grads_static[idx],
    #             self.out_grads_torch[idx],
    #             self.atol,
    #             self.rtol,
    #             self.dtype,
    #             version_a="paddle_develop",
    #             version_b="torch",
    #             eager_or_static_mode="static",
    #             fwd_or_bkd="backward",
    #             api="paddle.nn.functional.silu",
    #         )

    # def test_eager_stability(self):
    #     x_eager, dout_eager = self.gen_eager_inputs_and_dout()
    #     out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
    #         x_eager, dout_eager
    #     )
    #     out_eager_baseline_np = out_eager_baseline.numpy()
    #     out_grads_eager_baseline_np = map_structure(
    #         lambda x: x.numpy(),
    #         out_grads_eager_baseline,
    #     )
    #     del out_eager_baseline
    #     del out_grads_eager_baseline
    #     paddle.device.cuda.empty_cache()

    #     for i in range(50):
    #         out_eager, out_grads_eager = self.cal_eager_res(
    #             x_eager, dout_eager
    #         )
    #         out_eager = out_eager.numpy()
    #         out_grads_eager = map_structure(
    #             lambda x: x.numpy(),
    #             out_grads_eager,
    #         )
    #         # test develop eager forward stability
    #         np_assert_staility(
    #             out_eager,
    #             out_eager_baseline_np,
    #             self.dtype,
    #             version="paddle_develop",
    #             eager_or_static_mode="eager",
    #             fwd_or_bkd="forward",
    #             api="paddle.nn.functional.silu",
    #         )
    #         # test develop eager backward stability
    #         for idx in range(len(out_grads_eager)):
    #             np_assert_staility(
    #                 out_grads_eager[idx],
    #                 out_grads_eager_baseline_np[idx],
    #                 self.dtype,
    #                 version="paddle_develop",
    #                 eager_or_static_mode="eager",
    #                 fwd_or_bkd="backward",
    #                 api="paddle.nn.functional.silu",
    #             )

    # def test_static_stability(self):
    #     with paddle.fluid.framework._dygraph_guard(None):
    #         mp, sp = paddle.static.Program(), paddle.static.Program()
    #         with paddle.static.program_guard(mp, sp):
    #             (
    #                 x_static,
    #                 dout_static,
    #             ) = self.gen_static_inputs_and_dout()
    #             (out_static_pg, out_grads_static_pg) = self.cal_static_res(
    #                 x_static,
    #                 dout_static,
    #             )
    #         exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
    #         exe.run(sp)
    #         out = exe.run(
    #             mp,
    #             feed={"x": self.np_x, "dout": self.np_dout},
    #             fetch_list=[out_static_pg] + out_grads_static_pg,
    #         )
    #         out_static_baseline, out_grads_static_baseline = out[0], out[1:]
    #         for i in range(50):
    #             out = exe.run(
    #                 mp,
    #                 feed={"x": self.np_x, "dout": self.np_dout},
    #                 fetch_list=[out_static_pg] + out_grads_static_pg,
    #             )
    #             out_static, out_grads_static = out[0], out[1:]
    #             # test develop static forward stability
    #             np_assert_staility(
    #                 out_static,
    #                 out_static_baseline,
    #                 self.dtype,
    #                 version="paddle_develop",
    #                 eager_or_static_mode="static",
    #                 fwd_or_bkd="forward",
    #                 api="paddle.nn.functional.silu",
    #             )
    #             # test develop static backward stability
    #             for idx in range(len(out_grads_static)):
    #                 np_assert_staility(
    #                     out_grads_static[idx],
    #                     out_grads_static_baseline[idx],
    #                     self.dtype,
    #                     version="paddle_develop",
    #                     eager_or_static_mode="static",
    #                     fwd_or_bkd="backward",
    #                     api="paddle.nn.functional.silu",
    #                 )


class TestSquaredl2NormDevelopCase1_FP16(TestSquaredl2NormDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase1_BFP16(TestSquaredl2NormDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase2_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[12288, 32032]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase2_FP16(TestSquaredl2NormDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase2_BFP16(TestSquaredl2NormDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase3_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[12288, 64]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase3_FP16(TestSquaredl2NormDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase3_BFP16(TestSquaredl2NormDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase4_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[12288, 9216]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase4_FP16(TestSquaredl2NormDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase4_BFP16(TestSquaredl2NormDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase5_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[12288]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase5_FP16(TestSquaredl2NormDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase5_BFP16(TestSquaredl2NormDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase6_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1536, 12288]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase6_FP16(TestSquaredl2NormDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase6_BFP16(TestSquaredl2NormDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase7_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[1792]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase7_FP16(TestSquaredl2NormDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase7_BFP16(TestSquaredl2NormDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase8_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[32032, 12288]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase8_FP16(TestSquaredl2NormDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase8_BFP16(TestSquaredl2NormDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase9_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[32032]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase9_FP16(TestSquaredl2NormDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase9_BFP16(TestSquaredl2NormDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase10_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4608, 12288]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase10_FP16(TestSquaredl2NormDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase10_BFP16(TestSquaredl2NormDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestSquaredl2NormDevelopCase11_FP32(TestSquaredl2NormDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[9216]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

class TestSquaredl2NormDevelopCase11_FP16(TestSquaredl2NormDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "float16"

class TestSquaredl2NormDevelopCase11_BFP16(TestSquaredl2NormDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()

