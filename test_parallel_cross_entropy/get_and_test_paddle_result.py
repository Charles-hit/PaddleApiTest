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

dim = [1, 8192]

def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)

class TestPaddle(init_config_class.InitConfigClass):
    def __init__(self, group, id, np_input_dir="./inputs_case1.npz", dtype="float32", save_static_res_path="./static_develop_res_case1_float32.npz" , save_eager_res_path="./eager_develop_res_case1_float32.npz", torch_dir="1_torch_out_float32.npz"):
        self._init_params(np_input_dir, dtype, save_static_res_path, save_eager_res_path)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._group = group
        self.id =id
        world_size = paddle.distributed.get_world_size()
        rank = paddle.distributed.get_rank()
        self.np_paddle_logits = np.array_split(self.np_logits, world_size, axis = -1)[rank]
        if rank == 0:
            np_inputs_array = np.load(torch_dir)
            self._out_torch = np_inputs_array["torch_out"]
            self._out_grads_torch = np.array_split(np_inputs_array["torch_out_grad"], world_size, axis = -1)[rank]
        
    
    def _gen_eager_inputs_and_dout(self):
        place = paddle.device.get_device()
        logits_eager = paddle.to_tensor(
            self.np_paddle_logits,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place=place,
        )
        logits_eager.stop_gradient = False
        label_eager = paddle.to_tensor(
            self.np_label,
            dtype="int64",
            place=place,
        )
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place=place,
        )
        return logits_eager, label_eager, dout_eager

    def _gen_static_inputs_and_dout(self):
        logits_static = paddle.static.data(
            'logits',
            shape=self.np_paddle_logits.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        logits_static.stop_gradient = False
        label_static = paddle.static.data(
            'label',
            shape=self.np_label.shape,
            dtype="int64",
        )
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        return logits_static, label_static, dout_static

    def _cal_eager_res(self, logits, label, dout):
        logits_t = logits
        label_t = label
        dout_t = dout
        if self.dtype == "bfloat16":
            logits_t = paddle.cast(logits, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        
        loss_func = fleet.meta_parallel.ParallelCrossEntropy(mp_group=self._group, ignore_index=-100)
        out = loss_func(logits_t, label_t)

        out_grads = paddle.grad(
            [out], [logits_t], grad_outputs=[dout_t]
        )

        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = paddle.cast(out_grads[0], dtype="float32")

        return out, out_grads[0]

    def _cal_static_res(self, logits, label, dout):
        logits_t = logits
        label_t = label
        dout_t = dout
        if self.dtype == "bfloat16":
            logits_t = paddle.cast(logits, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")

        loss_func = fleet.meta_parallel.ParallelCrossEntropy(mp_group=self._group, ignore_index=-100)
        out = loss_func(logits_t, label_t)
        out = paddle.reshape(out, [dim[self.id]])
        out_grads = paddle.static.gradients(
            [out], [logits_t], target_gradients=[dout_t]
        )    

        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads =  paddle.cast(out_grads[0], dtype="float32")
        return out, out_grads

    def _test_eager_accuracy(self):
        logits_eager, label_eager, dout_eager = self._gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self._cal_eager_res(logits_eager, label_eager, dout_eager)

        del logits_eager
        del label_eager 
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager1 = paddle.reshape(out_eager, shape = [dim[self.id]])
        out_eager_np = out_eager1.numpy()
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
                    self.dtype,
                    version_a="paddle_develop",
                    version_b="torch",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="fleet.meta_parallel.ParallelCrossEntropy",
                )
            except Exception as e:
                print(e)
                print("eager_accuracy forward {dtype} failed".format(dtype=self.dtype))
            
            try:
                np_assert_accuracy(
                    out_grads_eager_np,
                    self._out_grads_torch,
                    self._atol,
                    self._rtol,
                    self.dtype,
                    version_a="paddle_develop",
                    version_b="torch",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="fleet.meta_parallel.ParallelCrossEntropy",
                )
            except Exception as e:
                print(e)
                print("eager_accuracy grad {dtype} failed".format(dtype=self.dtype))
        
    def _test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                logits_eager, label_eager, dout_eager = self._gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self._cal_static_res(
                    logits_eager,
                    label_eager,
                    dout_eager
                )
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"logits": self.np_paddle_logits,
                      "label": self.np_label, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
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
                    self.dtype,
                    version_a="paddle_develop",
                    version_b="torch",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="fleet.meta_parallel.ParallelCrossEntropy",
                )
            except Exception as e:
                print(e)
                print("static_accuracy forward {dtype} failed".format(dtype=self.dtype))

            try:
                np_assert_accuracy(
                    out_grads_static,
                    self._out_grads_torch,
                    self._atol,
                    self._rtol,
                    self.dtype,
                    version_a="paddle_develop",
                    version_b="torch",
                    eager_or_static_mode="static",
                    fwd_or_bkd="backward",
                    api="fleet.meta_parallel.ParallelCrossEntropy",
                )
            except Exception as e:
                print(e)
                print("static_accuracy grad {dtype} failed".format(dtype=self.dtype))


    def _test_eager_stability(self):
        logits_eager, label_eager, dout_eager = self._gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self._cal_eager_res(logits_eager, label_eager, dout_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = out_grads_eager_baseline.numpy()
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager, out_grads_eager = self._cal_eager_res(logits_eager, label_eager, dout_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = out_grads_eager.numpy()
                
            if paddle.distributed.get_rank() == 0:
                try: 
                    np_assert_staility(
                        out_eager,
                        out_eager_baseline_np,
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="eager",
                        fwd_or_bkd="forward",
                        api="fleet.meta_parallel.ParallelCrossEntropy",
                    )
                except Exception as e:
                    print(e)
                    print("eager_stability forward {dtype} failed".format(dtype=self.dtype))
                
                try:
                    np_assert_staility(
                        out_grads_eager,
                        out_grads_eager_baseline_np,
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="eager",
                        fwd_or_bkd="backward",
                        api="fleet.meta_parallel.ParallelCrossEntropy",
                    )
                except Exception as e:
                    print(e)
                    print("eager_stability grad {dtype} failed".format(dtype=self.dtype))

    def _test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                logits_eager, label_eager, dout_eager = self._gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self._cal_static_res(
                    logits_eager,
                    label_eager,
                    dout_eager,
                )
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"logits": self.np_paddle_logits,
                      "label": self.np_label, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"logits": self.np_paddle_logits,
                      "label": self.np_label, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]

                if paddle.distributed.get_rank() == 0:
                    try:
                        np_assert_staility(
                            out_static,
                            out_static_baseline,
                            self.dtype,
                            version="paddle_develop",
                            eager_or_static_mode="static",
                            fwd_or_bkd="forward",
                            api="fleet.meta_parallel.ParallelCrossEntropy",
                        )
                    except Exception as e:
                        print(e)
                        print("static_stability forward {dtype} failed".format(dtype=self.dtype))
                        
                    try: 
                        np_assert_staility(
                            out_grads_static,
                            out_grads_static_baseline,
                            self.dtype,
                            version="paddle_develop",
                            eager_or_static_mode="static",
                            fwd_or_bkd="backward",
                            api="fleet.meta_parallel.ParallelCrossEntropy",
                        )
                    except Exception as e:
                        print(e)
                        print("static_stability grad {dtype} failed".format(dtype=self.dtype))

dtype_list = ["float32"]

dist_strategy = fleet.DistributedStrategy()
world_size = paddle_dist.get_world_size()
dist_strategy.hybrid_configs = {
    "mp_degree": world_size,
    "pp_degree": 1,
    "dp_degree": 1,
}
paddle_dist.fleet.init(is_collective=True, strategy = dist_strategy)

set_random_seed(1024)

group = paddle_dist.new_group([i for i in range(world_size)], backend='nccl')

for id in [1, 2]:
    for dtype in dtype_list:

        np_input_dir = "./inputs_case{id}.npz".format(id=id)
        save_static_res_path = "./static_develop_res_case{id}_{dtype}.npz".format(id=id, dtype=dtype) 
        save_eager_res_path = "./eager_develop_res_case{id}_{dtype}.npz".format(id=id, dtype=dtype)
        torch_dir = "./torch_out_{dtype}_{id}.npz".format(dtype=dtype, id=id)

        test_paddle = TestPaddle(group, id - 1, np_input_dir, dtype, save_static_res_path, save_eager_res_path, torch_dir)
        test_paddle._test_eager_accuracy()
        print("eager {dtype} success".format(dtype=dtype))
        test_paddle._test_static_accuracy()
        print("static {dtype} success".format(dtype=dtype))
        test_paddle._test_eager_stability()
        print("eager_stability {dtype}  success".format(dtype=dtype))
        test_paddle._test_static_stability()
        print("static_stability {dtype}  success".format(dtype=dtype))

        print("{dtype} success".format(dtype=dtype))
