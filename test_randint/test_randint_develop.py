import sys
import unittest

import numpy as np

import paddle

sys.path.append("..")


class TestRandintDevelopCase1_INT64(unittest.TestCase):
    def setUp(self):
        self.init_params()

    def init_params(self):
        self.low = 0
        self.high = 1000
        self.size = [10, ]
        # default dtype is int64
        self.dtype = "int64"

    def cal_eager_res(self):
        paddle.seed(0)
        out = paddle.randint(low=self.low, high=self.high, shape=self.size, dtype=self.dtype)
        return out
    

    def test_eager_accuracy(self):
        out_eager = self.cal_eager_res()
        out_eager_np = out_eager.numpy()
        del out_eager
        paddle.device.cuda.empty_cache()
        np.testing.assert_equal(np.all(out_eager_np==0), False)
    
class TestRandintDevelopCase1_INT32(TestRandintDevelopCase1_INT64):
    def init_params(self):
        self.low = 0
        self.high = 1000
        self.size = [10, ]
        self.dtype = "int32"


class TestRandintDevelopCase2_INT64(TestRandintDevelopCase1_INT64):
    def init_params(self):
        self.low = 0
        self.high = 1000
        self.size = [4096, 4096, 256]
        self.dtype = "int64"

class TestRandintDevelopCase2_INT32(TestRandintDevelopCase1_INT64):
    def init_params(self):
        self.low = 0
        self.high = 1000
        self.size = [4096, 4096, 256]
        self.dtype = "int32"

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()