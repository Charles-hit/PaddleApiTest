import numpy as np
import paddle
import paddle.distributed as paddle_dist
import paddle.distributed.fleet as fleet
import init_config_class
import random
import sys
sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)

class TestPaddle(init_config_class.InitConfigClass):
    def __init__(self, group, id, np_input_dir="", dtype="", save_static_res_path="" , save_eager_res_path="", torch_dir=""):
        self._init_params(np_input_dir, dtype, save_static_res_path, save_eager_res_path)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._group = group
        self.id = id
        world_size = paddle.distributed.get_world_size()
        rank = paddle.distributed.get_rank()
        self._np_paddle_table = np.array_split(self._np_table, world_size)[rank]
        if rank == 0:
            np_inputs_array = np.load(torch_dir)
            self._out_torch = np_inputs_array["torch_out"]
            self._out_grads_torch = np.array_split(np_inputs_array["torch_out_grad"], world_size)[rank]
        
    
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

    def _gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self._np_x.shape,
            dtype="int64",
        )
        x_static.stop_gradient = False
        table_static = paddle.static.data(
            'table',
            shape=self._np_paddle_table.shape,
            dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        )
        table_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self._np_dout.shape,
            dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, table_static, dout_static

    def _cal_eager_res(self, x, table, dout):
        x_t = x
        table_t = table
        dout_t = dout
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

        return out, out_grads

    def _cal_static_res(self, x, table, dout):
        x_t = x
        table_t = table
        dout_t = dout
        if self._dtype == "bfloat16":
            table_t = paddle.cast(table, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")

        embedding = fleet.meta_parallel.VocabParallelEmbedding(init_config_class.dim_1[self.id], init_config_class.dim_3[self.id], mp_group=self._group)
        paddle.assign(table_t, embedding.weight)
        out = embedding(x_t)

        out_grads = paddle.static.gradients(
            [out], [embedding.weight], target_gradients=[dout_t]
        )

        out_grads = out_grads[0]

        if self._dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads =  paddle.cast(out_grads, dtype="float32")

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

        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()

        if paddle.distributed.get_rank() == 0:

            np.savez(self._save_eager_res_path, out_eager=out_eager_np, out_grads_eager=out_grads_eager_np)

            # compare eager res with torch
            try:
                np_assert_accuracy(
                    out_eager_np,
                    self._out_torch,
                    self._atol,
                    self._rtol,
                    self._dtype,
                    version_a="paddle_develop",
                    version_b="torch",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="fleet.meta_parallel.VocabParallelEmbedding",
                )
            except Exception as e:
                print(e)
                print("eager_accuracy forward {dtype} failed".format(dtype=self._dtype))
            try:
                np_assert_accuracy(
                    out_grads_eager_np,
                    self._out_grads_torch,
                    self._atol,
                    self._rtol,
                    self._dtype,
                    version_a="paddle_develop",
                    version_b="torch",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="fleet.meta_parallel.VocabParallelEmbedding",
                )
            except Exception as e:
                print(e)
                print("eager_accuracy backward {dtype} failed".format(dtype=self._dtype))
        
    def _test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, table_static, dout_static = self._gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self._cal_static_res(
                    x_static,
                    table_static,
                    dout_static,
            )
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self._np_x,  "table": self._np_paddle_table, "dout": self._np_dout},
                fetch_list=[out_static] + [out_grads_static],
            )
            out_static, out_grads_static = out[0], out[1:]
            out_grads_static = out_grads_static[0]
        
        if paddle.distributed.get_rank() == 0:

            np.savez(self._save_static_res_path, out_static=out_static, out_grads_static=out_grads_static)

            # compare static res with torch
            try:
                np_assert_accuracy(
                    out_static,
                    self._out_torch,
                    self._atol,
                    self._rtol,
                    self._dtype,
                    version_a="paddle_develop",
                    version_b="torch",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="fleet.meta_parallel.VocabParallelEmbedding",
                )
            except Exception as e:
                print(e)
                print("static_accuracy forward {dtype} failed".format(dtype=self._dtype))

            try:
                np_assert_accuracy(
                    out_grads_static,
                    self._out_grads_torch,
                    self._atol,
                    self._rtol,
                    self._dtype,
                    version_a="paddle_develop",
                    version_b="torch",
                    eager_or_static_mode="static",
                    fwd_or_bkd="backward",
                    api="fleet.meta_parallel.VocabParallelEmbedding",
                )
            except Exception as e:
                print(e)
                print("static_accuracy backward {dtype} failed".format(dtype=self._dtype))


    def _test_eager_stability(self):
        x_eager, table_eager, dout_eager = self._gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self._cal_eager_res(x_eager, table_eager, dout_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = out_grads_eager_baseline.numpy()
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(5):
            out_eager, out_grads_eager = self._cal_eager_res(x_eager, table_eager, dout_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = out_grads_eager.numpy()
                
            if paddle.distributed.get_rank() == 0:
                try:
                    np_assert_staility(
                        out_eager,
                        out_eager_baseline_np,
                        self._dtype,
                        version="paddle_develop",
                        eager_or_static_mode="eager",
                        fwd_or_bkd="forward",
                        api="fleet.meta_parallel.VocabParallelEmbedding",
                    )
                except Exception as e:
                    print(e)
                    print("eager_stability forward {dtype} failed".format(dtype=self._dtype))
                
                try:
                    np_assert_staility(
                        out_grads_eager,
                        out_grads_eager_baseline_np,
                        self._dtype,
                        version="paddle_develop",
                        eager_or_static_mode="eager",
                        fwd_or_bkd="backward",
                        api="fleet.meta_parallel.VocabParallelEmbedding",
                    )
                except Exception as e:
                    print(e)
                    print("eager_stability backward {dtype} failed".format(dtype=self._dtype))

    def _test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, table_static, dout_static = self._gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self._cal_static_res(
                    x_static,
                    table_static,
                    dout_static,
                )
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self._np_x, "table": self._np_paddle_table, "dout": self._np_dout},
                fetch_list=[out_static_pg] + [out_grads_static_pg],
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            
            for i in range(5):
                out = exe.run(
                    mp,
                    feed={"x": self._np_x, "table": self._np_paddle_table, "dout": self._np_dout},
                    fetch_list=[out_static_pg] + [out_grads_static_pg],
                )
                out_static, out_grads_static = out[0], out[1:]

                if paddle.distributed.get_rank() == 0:
                    try:
                        np_assert_staility(
                            out_static,
                            out_static_baseline,
                            self._dtype,
                            version="paddle_develop",
                            eager_or_static_mode="static",
                            fwd_or_bkd="forward",
                            api="fleet.meta_parallel.VocabParallelEmbedding",
                        )
                    except Exception as e:
                        print(e)
                        print("static_stability forward {dtype} failed".format(dtype=self._dtype))
                    try:
                        np_assert_staility(
                            out_grads_static[0],
                            out_grads_static_baseline[0],
                            self._dtype,
                            version="paddle_develop",
                            eager_or_static_mode="static",
                            fwd_or_bkd="backward",
                            api="fleet.meta_parallel.VocabParallelEmbedding",
                        )
                    except Exception as e:
                        print(e)
                        print("static_stability backward {dtype} failed".format(dtype=self._dtype))

dtype_list = ["float32", "float16", "bfloat16"]

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

for id in [1, 2]:
    for dtype in dtype_list:

        np_input_dir = "./inputs_case{id}.npz".format(id=id)
        save_static_res_path = "./static_develop_res_case{id}_{dtype}.npz".format(id=id, dtype=dtype) 
        save_eager_res_path = "./eager_develop_res_case{id}_{dtype}.npz".format(id=id, dtype=dtype)
        torch_dir = "./torch_out_{dtype}_{id}.npz".format(dtype=dtype, id=id)

        test_paddle = TestPaddle(group, id - 1, np_input_dir, dtype, save_static_res_path, save_eager_res_path, torch_dir)
        test_paddle._test_eager_accuracy()
        print("eager {dtype} finish".format(dtype=dtype))
        test_paddle._test_static_accuracy()
        print("static {dtype} finish".format(dtype=dtype))
        test_paddle._test_eager_stability()
        print("eager_stability {dtype}  finish".format(dtype=dtype))
        test_paddle._test_static_stability()
        print("static_stability {dtype}  finish".format(dtype=dtype))

        print("{dtype} finish".format(dtype=dtype))

