import sys
import unittest

import numpy as np

import paddle

sys.path.append("..")

class TestUniformDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()

    def init_params(self):
        self.size = [4096, 4096]
        self.low = -0.027063293868263706
        self.high = 0.027063293868263706
        self.seed = 0
        self.dtype = 'float32'

    def cal_eager_res(self):
        paddle.seed(0)
        out = paddle.uniform(shape=self.size, dtype=self.dtype, min=self.low, max=self.high, seed=self.seed)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        out_eager = self.cal_eager_res()
        out_eager_np = out_eager.numpy()
        del out_eager
        paddle.device.cuda.empty_cache()
        np.testing.assert_equal(np.all(out_eager_np==0), False)


class TestUniformDevelopCase1_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 4096]
        self.low = -0.027063293868263706
        self.high = 0.027063293868263706
        self.seed = 0
        self.dtype = 'float16'


class TestUniformDevelopCase1_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 4096]
        self.low = -0.027063293868263706
        self.high = 0.027063293868263706
        self.seed = 0
        self.dtype = 'bfloat16'




class TestUniformDevelopCase2_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 4096]
        self.low = -0.032639560491693344
        self.high = 0.032639560491693344
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase2_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 4096]
        self.low = -0.032639560491693344
        self.high = 0.032639560491693344
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase2_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 4096]
        self.low = -0.032639560491693344
        self.high = 0.032639560491693344
        self.seed = 0
        self.dtype = 'bfloat16'



class TestUniformDevelopCase3_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 12288]
        self.low = -0.019136638615493577
        self.high = 0.019136638615493577
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase3_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 12288]
        self.low = -0.019136638615493577
        self.high = 0.019136638615493577
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase3_BF16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 12288]
        self.low = -0.019136638615493577
        self.high = 0.019136638615493577
        self.seed = 0
        self.dtype = 'bfloat16'




class TestUniformDevelopCase4_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 1536]
        self.low = -0.032639560491693344
        self.high = 0.032639560491693344
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase4_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 1536]
        self.low = -0.032639560491693344
        self.high = 0.032639560491693344
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase4_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 1536]
        self.low = -0.032639560491693344
        self.high = 0.032639560491693344
        self.seed = 0
        self.dtype = 'bfloat16'



class TestUniformDevelopCase5_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [16384, 4096]
        self.low = -0.01711632992203644
        self.high = 0.01711632992203644
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase5_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [16384, 4096]
        self.low = -0.01711632992203644
        self.high = 0.01711632992203644
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase5_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [16384, 4096]
        self.low = -0.01711632992203644
        self.high = 0.01711632992203644
        self.seed = 0
        self.dtype = 'bfloat16'



class TestUniformDevelopCase6_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [3072, 1536]
        self.low = -0.03608439182435161
        self.high = 0.03608439182435161
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase6_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [3072, 1536]
        self.low = -0.03608439182435161
        self.high = 0.03608439182435161
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase6_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [3072, 1536]
        self.low = -0.03608439182435161
        self.high = 0.03608439182435161
        self.seed = 0
        self.dtype = 'bfloat16'




class TestUniformDevelopCase7_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 16384]
        self.low = -0.01711632992203644
        self.high = 0.01711632992203644
        self.seed = 0
        self.dtype = 'float32'


class TestUniformDevelopCase7_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 16384]
        self.low = -0.01711632992203644
        self.high = 0.01711632992203644
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase7_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 16384]
        self.low = -0.01711632992203644
        self.high = 0.01711632992203644
        self.seed = 0
        self.dtype = 'bfloat16'


class TestUniformDevelopCase8_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [512, 512]
        self.low = -0.07654655446197431
        self.high = 0.07654655446197431
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase8_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [512, 512]
        self.low = -0.07654655446197431
        self.high = 0.07654655446197431
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase8_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [512, 512]
        self.low = -0.07654655446197431
        self.high = 0.07654655446197431
        self.seed = 0
        self.dtype = 'bfloat16'



class TestUniformDevelopCase9_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 64]
        self.low = -0.06123724356957945
        self.high = 0.06123724356957945
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase9_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 64]
        self.low = -0.06123724356957945
        self.high = 0.06123724356957945
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase9_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 64]
        self.low = -0.06123724356957945
        self.high = 0.06123724356957945
        self.seed = 0
        self.dtype = 'bfloat16'



class TestUniformDevelopCase10_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 1536]
        self.low = -0.04419417382415922
        self.high = 0.04419417382415922
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase10_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 1536]
        self.low = -0.04419417382415922
        self.high = 0.04419417382415922
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase10_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 1536]
        self.low = -0.04419417382415922
        self.high = 0.04419417382415922
        self.seed = 0
        self.dtype = 'bfloat16'


class TestUniformDevelopCase11_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 4608]
        self.low = -0.03125
        self.high = 0.03125
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase11_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 4608]
        self.low = -0.03125
        self.high = 0.03125
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase11_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1536, 4608]
        self.low = -0.03125
        self.high = 0.03125
        self.seed = 0
        self.dtype = 'bfloat16'


class TestUniformDevelopCase12_FP32(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 4096, 256]
        self.low = -0.03125
        self.high = 0.03125
        self.seed = 0
        self.dtype = 'float32'

class TestUniformDevelopCase12_FP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 4096, 256]
        self.low = -0.03125
        self.high = 0.03125
        self.seed = 0
        self.dtype = 'float16'

class TestUniformDevelopCase12_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [4096, 4096, 256]
        self.low = -0.03125
        self.high = 0.03125
        self.seed = 0
        self.dtype = 'bfloat16'


if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()