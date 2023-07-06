import numpy as np
import paddle
import paddle.distributed as paddle_dist
import paddle.distributed.fleet as fleet
import init_config_class
import random
import sys
sys.path.append("../..")
from utils import (
    np_assert_accuracy
)

global_out = []
global_dout = []

def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)

class TestPaddle(init_config_class.InitConfigClass):
    def __init__(self, group, id, test_mode=1, np_input_dir="", dtype="", save_static_res_path="" , save_eager_res_path="", torch_dir=""):
        self._init_params(np_input_dir, dtype, save_static_res_path, save_eager_res_path)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._group = group
        self.id = id
        self.test_mode = test_mode
        world_size = paddle.distributed.get_world_size()
        rank = paddle.distributed.get_rank()
        print("self._np_table.shape: ", self._np_table.shape)
        self._np_paddle_table = np.array_split(self._np_table, world_size)[rank]
        if test_mode == 1:
            self._atol = 1e-2
        elif test_mode == 2:
            self._atol = 1e-6
    
    def _gen_eager_inputs_and_dout(self):
        place = paddle.device.get_device()
        x_eager = paddle.to_tensor(
            self._np_x,
            dtype="int64",
            place=place,
        )
        x_eager.stop_gradient = False
        table_eager = paddle.to_tensor(
            self._np_paddle_table,
            dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
            place=place,
        )
        table_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self._np_dout,
            dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
            place=place,
        )
        dout_eager.stop_gradient = False
        return x_eager, table_eager, dout_eager

    def _cal_eager_res(self, x, table, dout):
        x_t = x
        table_t = table
        dout_t = dout
        
        if self._dtype == "float32":
            table_t = paddle.cast(table, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
            table_t = paddle.cast(table, dtype="float32")
            dout_t = paddle.cast(dout, dtype="float32")
    
        if self._dtype == "bfloat16":
            table_t = paddle.cast(table, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")

        embedding = fleet.meta_parallel.VocabParallelEmbedding(init_config_class.dim_1[self.id], init_config_class.dim_3[self.id], mp_group=self._group)
        paddle.assign(table_t, embedding.weight)
        out = embedding(x_t)

        out_grads = paddle.grad(
            [out], [embedding.weight], grad_outputs=[dout_t]
        )

        out_grads = out_grads[0]

        if self._dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = paddle.cast(out_grads, dtype="float32")
        
        if self.test_mode == 2 and self._dtype == "float32":
            out = paddle.cast(out, dtype="uint16")
            out_grads = paddle.cast(out_grads, dtype="uint16")
            out = paddle.cast(out, dtype="float32")
            out_grads = paddle.cast(out_grads, dtype="float32")

        return out, out_grads

    def _test_eager_accuracy(self):
        x_eager, table_eager, dout_eager = self._gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self._cal_eager_res(x_eager, table_eager, dout_eager)

        del x_eager
        del table_eager 
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = out_grads_eager.numpy()

        global_out.append(out_eager_np)
        global_dout.append(out_grads_eager_np)

        if(self._dtype == "bfloat16"):
            try:
                np_assert_accuracy(
                    global_out[0],
                    global_out[1],
                    self._atol,
                    self._atol,
                    "fp32_vs_bf16",
                    version_a="fp32",
                    version_b="bf16",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="fleet.meta_parallel.VocabParallelEmbedding",
                )
            except Exception as e:
                print(e)

            try:
                np_assert_accuracy(
                    global_dout[0],
                    global_dout[1],
                    self._atol,
                    self._atol,
                    "fp32_vs_bf16",
                    version_a="fp32",
                    version_b="bf16",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="fleet.meta_parallel.VocabParallelEmbedding",
                )
            except Exception as e:
                print(e)

        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()


dtype_list = ["float32", "bfloat16"]

dist_strategy = fleet.DistributedStrategy()
world_size = paddle_dist.get_world_size()
dist_strategy.hybrid_configs = {
    "mp_degree": world_size,
    "pp_degree": 1,
    "dp_degree": 1,
}
paddle_dist.fleet.init(is_collective=True, strategy = dist_strategy)

set_random_seed(1024)

group = paddle_dist.collective._get_default_group()

for test_mode in [1,2]:
    print("test_mode_{test_mode} start*************************************************************************" \
        .format(test_mode=test_mode))
    for id in [1]:
        global_out.clear()
        global_dout.clear()   
        for dtype in dtype_list:

            np_input_dir = "./inputs_case{id}.npz".format(id=id)
            save_static_res_path = "./static_develop_res_case{id}_{dtype}.npz".format(id=id, dtype=dtype) 
            save_eager_res_path = "./eager_develop_res_case{id}_{dtype}.npz".format(id=id, dtype=dtype)
            torch_dir = "./torch_out_{dtype}_{id}.npz".format(dtype=dtype, id=id)

            test_paddle = TestPaddle(group, id - 1, test_mode, np_input_dir, dtype, save_static_res_path, save_eager_res_path, torch_dir)
            test_paddle._test_eager_accuracy()
            print("eager {dtype} finish".format(dtype=dtype))
            print("{dtype} finish".format(dtype=dtype))
    print("test_mode_{test_mode} end*************************************************************************" \
        .format(test_mode=test_mode))

