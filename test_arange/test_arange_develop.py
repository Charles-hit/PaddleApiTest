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

class TestArangeDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        out_torch = self.cal_torch_res()
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.torch_dtype = torch.float32
        self.start = 0 
        self.end = 128 
        self.step = 2 

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]


    def cal_torch_res(self):
        out = torch.arange(start=self.start, end=self.end, step=self.step, dtype=self.torch_dtype)
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out

    def cal_eager_res(self):
        out = paddle.arange(start=self.start, end=self.end, step=self.step, dtype=self.dtype)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def cal_static_res(self):
        out = paddle.arange(start=self.start, end=self.end, step=self.step, dtype=self.dtype)
        
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        out_eager = self.cal_eager_res()
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        del out_eager
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
            api="paddle.arange",
        )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp):
                out_static = self.cal_static_res()
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            out = exe.run(
                mp,
                fetch_list=[out_static],
            )
            out_static = out[0]

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
            api="paddle.arange",
        )

    def test_eager_stability(self):
        out_eager_baseline = self.cal_eager_res()
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager = self.cal_eager_res()
            out_eager = out_eager.numpy()
            # test develop eager forward stability
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.arange",
            )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp = paddle.static.Program()
            with paddle.static.program_guard(mp):
                out_static_pg = self.cal_static_res()
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            out = exe.run(
                mp,
                fetch_list=[out_static_pg],
            )
            out_static_baseline = out[0]
            for i in range(5):
                out = exe.run(
                    mp,
                    fetch_list=[out_static_pg],
                )
                out_static = out[0]
                # test develop static forward stability
                np_assert_staility(
                    out_static,
                    out_static_baseline,
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="paddle.arange",
                )


class TestArangeDevelopCase1_FP16(TestArangeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.torch_dtype = torch.float16
        self.start = 0 
        self.end = 128 
        self.step = 2 


class TestArangeDevelopCase1_BFP16(TestArangeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.torch_dtype = torch.bfloat16
        self.start = 0 
        self.end = 128 
        self.step = 2 


class TestArangeDevelopCase2_FP32(TestArangeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.torch_dtype = torch.float32
        self.start = 0 
        self.end = 4096
        self.step = 1 


class TestArangeDevelopCase2_FP16(TestArangeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.torch_dtype = torch.float16
        self.start = 0
        self.end = 4096
        self.step = 1


class TestArangeDevelopCase2_BFP16(TestArangeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.torch_dtype = torch.bfloat16
        self.start = 0
        self.end = 4096
        self.step = 1


class TestArangeDevelopCase3_FP32(TestArangeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.torch_dtype = torch.float32
        self.start = 0 
        self.end =128 
        self.step = 2 


class TestArangeDevelopCase3_FP16(TestArangeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.torch_dtype = torch.float16
        self.start = 0
        self.end = 128
        self.step = 2


class TestArangeDevelopCase3_BFP16(TestArangeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.torch_dtype = torch.bfloat16
        self.start = 0
        self.end =128
        self.step =2

class TestArangeDevelopCase4_FP32(TestArangeDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.torch_dtype = torch.float32
        self.start = 0 
        self.end =8192 
        self.step = 1 


class TestArangeDevelopCase4_FP16(TestArangeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.torch_dtype = torch.float16
        self.start = 0
        self.end = 8192
        self.step = 1


class TestArangeDevelopCase4_BFP16(TestArangeDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.torch_dtype = torch.bfloat16
        self.start = 0
        self.end = 8192
        self.step = 1

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()

