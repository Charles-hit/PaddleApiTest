import sys
import unittest
import gc

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

class TestConcatDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, dout_torch
        )
        for i in range(self.data_num):
            del x_torch[0]
        del x_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().detach().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        gc.collect()
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        self.data_num = 64
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(64):
            x = np.random.random(size=[1, 4096, 4096]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[64, 4096, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(64):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def gen_torch_inputs_and_dout(self):
        x_torch = []
        for i in range(self.data_num):
            x = torch.tensor(
                self.np_x[i],
                device='cuda',
                dtype=convert_dtype_to_torch_type(self.dtype)
                if self.dtype != 'bfloat16'
                else torch.float32,
                requires_grad=True,
            )
            x_torch.append(x)
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
        x_eager = []
        for i in range(self.data_num):
            x = paddle.to_tensor(
                self.np_x[i],
                dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
                place="gpu",
            )
            x.stop_gradient = False
            x_eager.append(x)
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = []
        for i in range(self.data_num):
            x = paddle.static.data(
                'x_%d' % i,
                shape=self.np_x[i].shape,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            )
            x.stop_gradient = False
            x_static.append(x)
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, dout_static

    def cal_torch_res(self, x, dout):
        if self.dtype == "bfloat16":
            for i in range(self.data_num):
                x[i] = x[i].to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        out = torch.cat(x, dim=self.axis)
        out_grads = torch.autograd.grad([out], x, grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, dout):
        if self.dtype == "bfloat16":
            for i in range(self.data_num):
                x[i] = paddle.cast(x[i], dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.concat(x, axis=self.axis)
        out_grads = paddle.grad([out], x, grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def cal_static_res(self, x, dout):
        if self.dtype == "bfloat16":
            for i in range(self.data_num):
                x[i] = paddle.cast(x[i], dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.concat(x, axis=self.axis)
        out_grads = paddle.static.gradients(
            [out], x, target_gradients=[dout]
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
        for i in range(self.data_num):
            del x_eager[0]
        del x_eager
        del dout_eager
        gc.collect()
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        del out_eager
        del out_grads_eager
        gc.collect()
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
            api="paddle.concat",
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
                api="paddle.concat",
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            feed = {}
            for i in range(self.data_num):
                feed["x_%d" % i] = self.np_x[i]
            feed["dout"] = self.np_dout
            out = exe.run(
                mp,
                feed=feed,
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # compare develop static forward res with torch
        np_assert_accuracy(
            out_static,
            self.out_torch,
            self.atol,
            self.rtol,
            self.dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="static",
            fwd_or_bkd="forward",
            api="paddle.concat",
        )
        # compare develop static backward res with torch
        for idx in range(len(out_grads_static)):
            np_assert_accuracy(
                out_grads_static[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="backward",
                api="paddle.concat",
            )

    def test_eager_stability(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, dout_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager_baseline,
        )
        del out_eager_baseline
        del out_grads_eager_baseline
        gc.collect()
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager, out_grads_eager = self.cal_eager_res(
                x_eager, dout_eager
            )
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                lambda x: x.numpy(),
                out_grads_eager,
            )
            # test develop eager forward stability
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.concat",
            )
            # test develop eager backward stability
            for idx in range(len(out_grads_eager)):
                np_assert_staility(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="paddle.concat",
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            feed = {}
            for i in range(self.data_num):
                feed["x_%d" % i] = self.np_x[i]
            feed["dout"] = self.np_dout
            out = exe.run(
                mp,
                feed=feed,
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(5):
                out = exe.run(
                    mp,
                    feed=feed,
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                # test develop static forward stability
                np_assert_staility(
                    out_static,
                    out_static_baseline,
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="paddle.concat",
                )
                # test develop static backward stability
                for idx in range(len(out_grads_static)):
                    np_assert_staility(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="backward",
                        api="paddle.concat",
                    )


class TestConcatDevelopCase1_FP16(TestConcatDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase1_BFP16(TestConcatDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase2_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 64
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(64):
            x = np.random.random(size=[1, 4096]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[64, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(64):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase2_FP16(TestConcatDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase2_BFP16(TestConcatDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase3_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 64
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(64):
            x = np.random.random(size=[1]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[64]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(64):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase3_FP16(TestConcatDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase3_BFP16(TestConcatDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase4_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 7
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(7):
            x = np.random.random(size=[1]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[7]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(7):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase4_FP16(TestConcatDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase4_BFP16(TestConcatDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase5_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 6
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(6):
            x = np.random.random(size=[1]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[6]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(6):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase5_FP16(TestConcatDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase5_BFP16(TestConcatDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase6_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 64
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(64):
            x = np.random.random(size=[4096, 1]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[262144, 1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(64):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase6_FP16(TestConcatDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase6_BFP16(TestConcatDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase7_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 2
        self.axis = -1
        # init np array
        self.np_x = []
        for i in range(2):
            x = np.random.random(size=[4096, 64]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[4096, 128]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(2):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase7_FP16(TestConcatDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase7_BFP16(TestConcatDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase8_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 6
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(6):
            x = np.random.random(size=[4096]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[24576]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(6):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase8_FP16(TestConcatDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase8_BFP16(TestConcatDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase9_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 64
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(64):
            x = np.random.random(size=[4224, 1]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[270336, 1]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(64):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase9_FP16(TestConcatDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase9_BFP16(TestConcatDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase10_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 6
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(6):
            x = np.random.random(size=[8192]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[49152]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(6):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase10_FP16(TestConcatDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase10_BFP16(TestConcatDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase11_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 64
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(64):
            x = np.random.random(size=[1, 4224]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[64, 4224]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(64):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase11_FP16(TestConcatDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase11_BFP16(TestConcatDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestConcatDevelopCase12_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 12
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(self.data_num):
            x = np.random.random(size=[14336]).astype("float32") - 0.5
            self.np_x.append(x)
        self.np_dout = np.random.random(size=[self.data_num*14336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(self.data_num):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase12_FP16(TestConcatDevelopCase12_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase12_BFP16(TestConcatDevelopCase12_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestConcatDevelopCase13_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 6
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(5):
            x = np.random.random(size=[14336]).astype("float32") - 0.5
            self.np_x.append(x)

        x = np.random.random(size=[77070336]).astype("float32") - 0.5
        self.np_x.append(x)            
        self.np_dout = np.random.random(size=[5*14336 + 77070336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(self.data_num):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase13_FP16(TestConcatDevelopCase13_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase13_BFP16(TestConcatDevelopCase13_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestConcatDevelopCase14_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 5
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(self.data_num):
            x = np.random.random(size=[14336]).astype("float32") - 0.5
            self.np_x.append(x)

        self.np_dout = np.random.random(size=[self.data_num*14336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(self.data_num):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase14_FP16(TestConcatDevelopCase14_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase14_BFP16(TestConcatDevelopCase14_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestConcatDevelopCase15_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 1
        self.axis = 0
        # init np array
        self.np_x = []
        for i in range(self.data_num):
            x = np.random.random(size=[179601408]).astype("float32") - 0.5
            self.np_x.append(x)

        self.np_dout = np.random.random(size=[self.data_num*179601408]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(self.data_num):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase15_FP16(TestConcatDevelopCase15_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase15_BFP16(TestConcatDevelopCase15_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestConcatDevelopCase16_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 3
        self.axis = 0
        # init np array
        self.np_x = []
        x = np.random.random(size=[5376]).astype("float32") - 0.5
        self.np_x.append(x)
        x = np.random.random(size=[25690112]).astype("float32") - 0.5
        self.np_x.append(x)
        x = np.random.random(size=[14336]).astype("float32") - 0.5
        self.np_x.append(x)

        self.np_dout = np.random.random(size=[5376 + 25690112 + 14336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(self.data_num):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase16_FP16(TestConcatDevelopCase16_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase16_BFP16(TestConcatDevelopCase16_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestConcatDevelopCase17_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 1
        self.axis = 0
        # init np array
        self.np_x = []
        x = np.random.random(size=[77070336]).astype("float32") - 0.5
        self.np_x.append(x)

        self.np_dout = np.random.random(size=[77070336]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(self.data_num):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase17_FP16(TestConcatDevelopCase17_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase17_BFP16(TestConcatDevelopCase17_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestConcatDevelopCase18_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 2
        self.axis = -1
        # init np array
        self.np_x = []
        for i in range(self.data_num):
            x = np.random.random(size=[8192, 64]).astype("float32") - 0.5
            self.np_x.append(x)

        self.np_dout = np.random.random(size=[8192, 64*self.data_num]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(self.data_num):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase18_FP16(TestConcatDevelopCase18_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase18_BFP16(TestConcatDevelopCase18_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

class TestConcatDevelopCase19_FP32(TestConcatDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        self.data_num = 2
        self.axis = 0
        # init np array
        self.np_x = []
        x = np.random.random(size=[9632]).astype("float32") - 0.5
        self.np_x.append(x)
        x = np.random.random(size=[69042176]).astype("float32") - 0.5
        self.np_x.append(x)

        self.np_dout = np.random.random(size=[69042176 + 9632]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            for i in range(self.data_num):
                self.np_x[i] = self.np_x[i].astype("float16")
            self.np_dout = self.np_dout.astype("float16")


class TestConcatDevelopCase19_FP16(TestConcatDevelopCase19_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestConcatDevelopCase19_BFP16(TestConcatDevelopCase19_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
