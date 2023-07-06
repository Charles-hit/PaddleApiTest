import numpy as np
import paddle
import paddle.distributed as paddle_dist
import paddle.distributed.fleet as fleet
import init_config_class
import random
import sys
sys.path.append("../../..")
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
    def __init__(self, group, test_mode=1, np_input_dir_forward="", np_input_dir_backward="", dtype=""):
        self._init_params(np_input_dir_forward, np_input_dir_backward, dtype)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._group = group
        self.test_mode = test_mode
    
    def _gen_eager_inputs_and_dout(self):
        place = paddle.device.get_device()
        x_eager = paddle.to_tensor(
            self._np_x,
            dtype="int64",
            place=place,
        )
        x_eager.stop_gradient = False
        table_eager = paddle.to_tensor(
            self._np_table,
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

        embedding = fleet.meta_parallel.VocabParallelEmbedding(self._num_embeddings, self._embedding_dim, mp_group=self._group)
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
rank = paddle_dist.get_rank()
paddle_dist.fleet.init(is_collective=True, strategy = dist_strategy)

set_random_seed(1024)

group = paddle_dist.collective._get_default_group()

for test_mode in [1,2]:
    print("test_mode_{test_mode} start*************************************************************************" \
        .format(test_mode=test_mode))

    if test_mode == 1:
        atol = 1e-2
    elif test_mode == 2:
        atol = 1e-6

    global_out.clear()
    global_dout.clear()   
    for dtype in dtype_list:

        np_input_dir_forward = "./data/vpe-int64-bf16-bf16-eager_in_tmp_477-word_embedding_expanded_{rank}.w_0-eager_in_tmp_486-pp-0-mp-{rank}.npz".format(rank=rank)
        np_input_dir_backward = "./data/vpe-int64-bf16-bf16-eager_in_tmp_477-word_embedding_expanded_{rank}.w_0-eager_in_tmp_486-pp-0-mp-{rank}.npy".format(rank=rank)

        test_paddle = TestPaddle(group, test_mode, np_input_dir_forward, np_input_dir_backward, dtype)
        test_paddle._test_eager_accuracy()
    
    try:
        np_assert_accuracy(
            global_out[0],
            global_out[1],
            atol,
            atol,
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
            atol,
            atol,
            "fp32_vs_bf16",
            version_a="fp32",
            version_b="bf16",
            eager_or_static_mode="eager",
            fwd_or_bkd="backward",
            api="fleet.meta_parallel.VocabParallelEmbedding",
        )
    except Exception as e:
        print(e)
    print("test_mode_{test_mode} end*************************************************************************" \
        .format(test_mode=test_mode))

