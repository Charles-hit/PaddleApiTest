import numpy as np
import torch
import torch.distributed as torch_dist
import init_config_class
import prepare_data
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

class TestTorch(init_config_class.InitConfigClass):
    def __init__(self, device, np_input_dir="./inputs_case1.npz", dtype="float32", torch_dir="1_torch_out_float32.npz"):
        self._init_params(np_input_dir, dtype)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._device = device
        logits_torch, label_torch, dout_torch = self._gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self._cal_torch_res(logits_torch, label_torch, dout_torch)
        del logits_torch 
        del label_torch 
        del dout_torch
        
        a = out_torch.cpu().detach().numpy()
        b = out_grads_torch.cpu().detach().numpy()
        np.savez(torch_dir, torch_out=a, torch_out_grad=b)
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()
    
    def _gen_torch_inputs_and_dout(self):
        logits_torch = torch.tensor(
            self.np_logits,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        label_torch = torch.tensor(
            self.np_label,
            device='cuda',
            dtype=torch.int64,
            requires_grad=False,
        )
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return logits_torch, label_torch, dout_torch

    def _cal_torch_res(self, logits, label, dout):
        logits_t = logits
        label_t = label
        dout_t = dout
        if self.dtype == "bfloat16":
            logits_t = logits.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        out = torch.nn.functional.cross_entropy(
            logits_t, label_t, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='none', label_smoothing=0.0)
        out_grads = torch.autograd.grad(
            [out], [logits_t], grad_outputs=[dout_t])
        out_grads = out_grads[0]
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = out_grads.to(dtype=torch.float32)
        return out, out_grads

dtype_list = ["float32"]

prepare_data.generate_np_inputs_and_dout()

for id in [1, 2]:
    for dtype in dtype_list:

        np_input_dir = "./inputs_case{id}.npz".format(id=id)
        torch_dir = "./torch_out_{dtype}_{id}.npz".format(dtype=dtype, id=id)

        test_torch = TestTorch('cuda', np_input_dir, dtype, torch_dir)