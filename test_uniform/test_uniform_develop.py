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


class TestUniformDevelopCase13_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [49500, 2048]
        self.low = -0.010788714864769802
        self.high = 0.010788714864769802
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase14_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5504, 2048]
        self.low = -0.02818672605010608
        self.high = 0.02818672605010608
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase15_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [2048, 49500]
        self.low = -0.010788714864769802
        self.high = 0.010788714864769802
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase16_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [2048, 2048]
        self.low = -0.038273277230987154
        self.high = 0.038273277230987154
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase17_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 192]
        self.low = -0.03360830362111656
        self.high = 0.03360830362111656
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase18_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [2048, 5504]
        self.low = -0.02818672605010608
        self.high = 0.02818672605010608
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase19_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 2048]
        self.low = -0.028931878117892232
        self.high = 0.028931878117892232
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase20_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 1920]
        self.low = -0.02919371040605711
        self.high = 0.02919371040605711
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase21_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1280, 1280]
        self.low = -0.04841229182759271
        self.high = 0.04841229182759271
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase22_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [1280, 5120]
        self.low = -0.030618621784789725
        self.high = 0.030618621784789725
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase23_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 1280]
        self.low = -0.030618621784789725
        self.high = 0.030618621784789725
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase24_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [640, 5120]
        self.low = -0.03227486121839514
        self.high = 0.03227486121839514
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase25_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [3200, 5120]
        self.low = -0.026854307776478733
        self.high = 0.026854307776478733
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase26_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [49408, 1280]
        self.low = -0.010879853497231116
        self.high = 0.010879853497231116
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase27_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 3200]
        self.low = -0.026854307776478733
        self.high = 0.026854307776478733
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase28_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 5120]
        self.low = -0.024206145913796356
        self.high = 0.024206145913796356
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase29_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 15360]
        self.low = -0.01711632992203644
        self.high = 0.01711632992203644
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase30_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 1536]
        self.low = -0.03002402883845384
        self.high = 0.03002402883845384
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase31_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [5120, 25600]
        self.low = -0.013975424859373685
        self.high = 0.013975424859373685
        self.seed = 0
        self.dtype = 'bfloat16'

class TestUniformDevelopCase32_BFP16(TestUniformDevelopCase1_FP32):
    def init_params(self):
        self.size = [25600, 5120]
        self.low = -0.013975424859373685
        self.high = 0.013975424859373685
        self.seed = 0
        self.dtype = 'bfloat16'



if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()