import numpy as np
import torch
import torch.distributed as torch_dist
import init_config_class
import prepare_data
import sys
sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

class TestTorch(init_config_class.InitConfigClass):
    def __init__(self, device, id, np_input_dir="", dtype="", torch_dir=""):
        self._init_params(np_input_dir, dtype)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._device = device
        self.id = id
        out_torch, out_grads_torch = self._get_and_compare_torch_result()
        
        np.savez(torch_dir, torch_out=out_torch, torch_out_grad=out_grads_torch)
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()
    
    def _gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self._np_x,
            device=self._device,
            dtype=convert_dtype_to_torch_type("int64"),
        )
        table_torch = torch.tensor(
            self._np_table,
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
        return x_torch, table_torch, dout_torch

    def _cal_torch_res(self, x, table, dout):
        x_t = x
        table_t = table
        dout_t = dout

        if self._dtype == "bfloat16":
            table_t = table_t.to(dtype=torch.bfloat16)
            dout_t = dout_t.to(dtype=torch.bfloat16)

        embedding = torch.nn.Embedding(init_config_class.dim_1[self.id], init_config_class.dim_3[self.id], _weight=table_t, dtype=self._dtype)
        out = embedding(x_t)
        out_grads = torch.autograd.grad([out], [embedding.weight], grad_outputs=[dout_t])
        out_grads = out_grads[0]

        if self._dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = out_grads.to(dtype=torch.float32)
        
        return out, out_grads
    
    def _get_and_compare_torch_result(self):
        x_torch, table_torch, dout_torch = self._gen_torch_inputs_and_dout()
        base_out, base_dout = self._cal_torch_res(x_torch, table_torch, dout_torch)
        base_out_np = base_out.detach().cpu().numpy()
        base_dout_np = base_dout.detach().cpu().numpy()
        # for i in range(5):
        #     x_torch, table_torch, dout_torch = self._gen_torch_inputs_and_dout()
        #     out, dout = self._cal_torch_res(x_torch, table_torch, dout_torch)
        #     out_np =  out.detach().cpu().numpy()
        #     dout_np = dout.detach().cpu().numpy()
        #     try:
        #         np_assert_staility(
        #             base_out_np,
        #             out_np,
        #             self._dtype,
        #             version="torch",
        #             eager_or_static_mode="eager",
        #             fwd_or_bkd="forward",
        #             api="torch.nn.Embedding",
        #         )
        #     except Exception as e:
        #         print(e)
        #         print("torch_stability forward {dtype} failed".format(dtype=self._dtype))
        #     try:
        #         np_assert_staility(
        #             base_dout_np,
        #             dout_np,
        #             self._dtype,
        #             version="torch",
        #             eager_or_static_mode="eager",
        #             fwd_or_bkd="backward",
        #             api="torch.nn.Embedding",
        #         )
        #     except Exception as e:
        #         print(e)
        #         print("torch_stability backward {dtype} failed".format(dtype=self._dtype))
        return base_out_np, base_dout_np

dtype_list = ["float32", "float16", "bfloat16"]

prepare_data.generate_np_inputs_and_dout()
for id in [1, 2]:
    for dtype in dtype_list:

        np_input_dir = "./inputs_case{id}.npz".format(id=id)
        torch_dir = "./torch_out_{dtype}_{id}.npz".format(dtype=dtype,id=id)

        test_torch = TestTorch('cuda', id - 1, np_input_dir, dtype, torch_dir)
