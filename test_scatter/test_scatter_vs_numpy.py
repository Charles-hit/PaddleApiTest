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

def expand_class(shapes, dtypes):
    def decorate(cls):
        test_cls_module = sys.modules[cls.__module__].__dict__
        unittest_num = 1
        for shape in shapes:
            for dtype in dtypes:
                test_cls = dict(cls.__dict__)
                test_cls["shapes"] = shape
                test_cls["dtype"] = dtype
                name = cls.__name__ + str(unittest_num)
                unittest_num += 1
                test_cls_module[name] = type(name, (cls,), test_cls)

        for m in list(cls.__dict__):
            if m.startswith("test"):
                delattr(cls, m)
        return cls

    return decorate

shape_all = [[[1035],[12],[12]],[[1061],[1061],[1061]],[[1099],[76],[76]],[[1379],[356],[356]],[[1039],[1039],[1039]],[[1041],[1041],[1041]],[[1040],[17],[17]],[[1057],[1057],[1057]],[[1055],[32],[32]],[[1064],[41],[41]],[[1077],[1077],[1077]],[[1054],[31],[31]],[[1056],[33],[33]],[[1064],[1064],[1064]],[[1044],[21],[21]],[[1056],[1056],[1056]],[[1054],[1054],[1054]],[[1065],[42],[42]],[[1042],[19],[19]],[[1059],[1059],[1059]],[[1041],[18],[18]],[[1047],[1047],[1047]],[[1053],[30],[30]],[[1052],[29],[29]],[[1076],[1076],[1076]],[[1034],[1034],[1034]],[[1061],[38],[38]],[[1078],[55],[55]],[[1040],[1040],[1040]],[[1035],[1035],[1035]],[[1078],[1078],[1078]],[[1038],[15],[15]],[[1043],[1043],[1043]],[[1058],[35],[35]],[[1036],[13],[13]],[[1058],[1058],[1058]],[[1049],[26],[26]],[[1069],[46],[46]],[[1050],[1050],[1050]],[[1046],[1046],[1046]],[[1045],[22],[22]],[[1069],[1069],[1069]],[[1038],[1038],[1038]],[[1062],[1062],[1062]],[[1050],[27],[27]],[[1051],[28],[28]],[[1048],[1048],[1048]],[[1034],[11],[11]],[[1037],[1037],[1037]],[[1090],[67],[67]],[[1055],[1055],[1055]],[[1059],[36],[36]],[[1090],[1090],[1090]],[[1045],[1045],[1045]],[[1057],[34],[34]],[[1067],[44],[44]],[[1039],[16],[16]],[[1037],[14],[14]],[[1053],[1053],[1053]],[[1042],[1042],[1042]],[[1036],[1036],[1036]],[[1052],[1052],[1052]],[[1047],[24],[24]],[[1077],[54],[54]],[[1071],[48],[48]],[[1065],[1065],[1065]],[[1049],[1049],[1049]],[[1379],[1379],[1379]],[[1043],[20],[20]],[[1066],[1066],[1066]],[[1076],[53],[53]],[[1066],[43],[43]],[[1067],[1067],[1067]],[[1062],[39],[39]],[[1099],[1099],[1099]],[[1051],[1051],[1051]],[[1044],[1044],[1044]],[[1046],[23],[23]],[[1048],[25],[25]],[[1071],[1071],[1071]],[[3041],[2018],[2018]],[[1071],[1071],[1071]],[[1043],[20],[20]],[[1057],[34],[34]],[[1050],[27],[27]],[[1066],[43],[43]],[[2913],[2913],[2913]],[[1081],[1081],[1081]],[[1038],[15],[15]],[[1045],[1045],[1045]],[[1061],[38],[38]],[[1050],[1050],[1050]],[[1048],[25],[25]],[[1052],[1052],[1052]],[[1048],[1048],[1048]],[[1058],[35],[35]],[[1039],[1039],[1039]],[[2913],[1890],[1890]],[[1044],[1044],[1044]],[[1046],[23],[23]],[[2377],[1354],[1354]],[[1067],[44],[44]],[[2983],[1960],[1960]],[[1038],[1038],[1038]],[[1034],[11],[11]],[[1078],[55],[55]],[[1049],[1049],[1049]],[[1036],[13],[13]],[[2287],[2287],[2287]],[[1077],[54],[54]],[[1040],[17],[17]],[[1057],[1057],[1057]],[[1054],[1054],[1054]],[[1077],[1077],[1077]],[[1051],[1051],[1051]],[[1052],[29],[29]],[[1047],[1047],[1047]],[[1041],[1041],[1041]],[[1063],[40],[40]],[[1055],[32],[32]],[[1034],[1034],[1034]],[[1043],[1043],[1043]],[[1054],[31],[31]],[[1040],[1040],[1040]],[[2955],[2955],[2955]],[[1047],[24],[24]],[[1036],[1036],[1036]],[[1037],[14],[14]],[[1067],[1067],[1067]],[[2377],[2377],[2377]],[[1070],[47],[47]],[[2339],[2339],[2339]],[[1072],[1072],[1072]],[[1046],[1046],[1046]],[[1062],[39],[39]],[[1072],[49],[49]],[[1061],[1061],[1061]],[[1078],[1078],[1078]],[[1039],[16],[16]],[[1059],[1059],[1059]],[[1063],[1063],[1063]],[[1035],[12],[12]],[[2983],[2983],[2983]],[[1070],[1070],[1070]],[[1041],[18],[18]],[[1042],[1042],[1042]],[[1060],[1060],[1060]],[[1056],[33],[33]],[[1037],[1037],[1037]],[[1042],[19],[19]],[[1035],[1035],[1035]],[[1058],[1058],[1058]],[[1081],[58],[58]],[[1053],[30],[30]],[[3041],[3041],[3041]],[[2745],[2745],[2745]],[[1045],[22],[22]],[[1053],[1053],[1053]],[[2745],[1722],[1722]],[[1059],[36],[36]],[[2287],[1264],[1264]],[[2339],[1316],[1316]],[[1055],[1055],[1055]],[[1071],[48],[48]],[[1062],[1062],[1062]],[[1049],[26],[26]],[[1051],[28],[28]],[[1060],[37],[37]],[[1056],[1056],[1056]],[[2955],[1932],[1932]],[[1044],[21],[21]],[[1066],[1066],[1066]]]
@expand_class(shapes = shape_all, dtypes=["float32","float16","bfloat16"])
class TestScatterDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        self.out_torch = np.copy(self.np_x)
        self.out_torch[self.np_index] = self.np_updates

    def init_params(self):
        self.x_shape = self.shapes[0]
        self.index_shape = self.shapes[1]
        self.updates_shape = self.shapes[2]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        assert 1 == len(self.x_shape)
        self.np_index = np.random.choice(self.x_shape[0], size=self.index_shape[0], replace=False)
        self.np_x = np.random.random(size=self.x_shape).astype("float32") - 0.5
        self.np_updates = np.random.random(size=self.index_shape).astype("float32") - 0.5
        self.np_dout = np.random.random(size=self.x_shape).astype("float32") - 0.5
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_updates = self.np_updates.astype("float16")
            self.np_dout = self.np_dout.astype("float16")


    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        index_eager = paddle.to_tensor(
            self.np_index,
            dtype="int64",
            place="gpu",
        )
        # index_eager.stop_gradient = False
        updates_eager = paddle.to_tensor(
            self.np_updates,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        updates_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, index_eager, updates_eager, dout_eager

    def cal_eager_res(self, x_eager, index_eager, updates_eager, dout_eager):
        out = paddle.scatter(x_eager, index_eager, updates_eager)
        return out

    def test_eager_accuracy(self):
        x_eager, index_eager, updates_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager = self.cal_eager_res(
            x_eager, index_eager, updates_eager, dout_eager
        )

        del x_eager
        del index_eager
        del updates_eager
        del dout_eager
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
            api="paddle.scatter",
        )

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()
