import numpy as np
import torch
import torch.distributed as torch_dist
import init_config_class
import sys
sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

class TestTorch(init_config_class.InitConfigClass):
    def __init__(self, group, device, np_input_dir="", dtype="", torch_dir=""):
        self._init_params(np_input_dir, dtype)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._group = group
        self._device = device
        out_torch, out_grads_torch = self._get_and_compare_torch_result()
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            np.savez(torch_dir, torch_out=out_torch, torch_out_grad=out_grads_torch)
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()
    
    def _gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self._np_x,
            device=self._device,
            dtype=convert_dtype_to_torch_type(self._dtype)
            if self._dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dout_torch = torch.tensor(
            self._np_dout,
            device=self._device,
            dtype=convert_dtype_to_torch_type(self._dtype)
            if self._dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, dout_torch

    def _cal_torch_res(self, x, dout):
        out = x
        dout_t = dout
        if self._dtype == "bfloat16":
            out = x.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        
        torch_dist.all_reduce(out, group=group)

        if self._dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            dout_t = dout_t.to(dtype=torch.float32)
        
        return out, dout_t

    def _get_and_compare_torch_result(self):
        x_torch, dout_torch = self._gen_torch_inputs_and_dout()
        base_out, base_dout = self._cal_torch_res(x_torch, dout_torch)
        base_out_np = base_out.detach().cpu().numpy()
        base_dout_np = base_dout.detach().cpu().numpy()
        for i in range(5):
            x_torch, dout_torch = self._gen_torch_inputs_and_dout()
            out, _ = self._cal_torch_res(x_torch, dout_torch)
            out_np =  out.detach().cpu().numpy()
            try:
                np_assert_staility(
                    base_out_np,
                    out_np,
                    self._dtype,
                    version="torch",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="torch.distributed.all_reduce",
                )
            except Exception as e:
                print(e)
                print("torch_stability forward {dtype} failed".format(dtype=self._dtype))
        return base_out_np, base_dout_np



dtype_list = ["float32", "float16", "bfloat16"]

torch_dist.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
world_size = torch_dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
group = torch_dist.new_group([i for i in range(world_size)])

for case_id in [1, 2, 3]:
    for dtype in dtype_list:

        np_input_dir = "./inputs_case{id}.npz".format(id=case_id)
        torch_dir = "{id}_torch_out_{dtype}.npz".format(id=case_id, dtype=dtype)

        test_torch = TestTorch(group ,device, np_input_dir, dtype, torch_dir)
