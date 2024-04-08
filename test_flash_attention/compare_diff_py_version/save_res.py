import torch
import numpy as np
import sys
import os
from paddle.utils import map_structure
import paddle
from paddle.nn.functional.flash_attention import flash_attention


sys.path.append("...")
from case_list import case_list
from utils import (
    convert_dtype_to_torch_type,

)


class SaveRes:

    def save_np_data(self, dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax):
        save_path = self.init_np_save_path(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)
        self.init_params(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)
        self.init_np_inputs_and_dout()
        np.save(
            save_path + '/q.npy',
            self.np_q
        )
        np.save(
            save_path + '/k.npy',
            self.np_k
        )
        np.save(
            save_path + '/v.npy',
            self.np_v
        )
        np.save(
            save_path + '/dout.npy',
            self.np_dout
        )


    def save_torch_res(self, dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax):
        save_path = self.init_torch_save_path(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)
        np_path = self.init_np_save_path(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)

        self.init_params(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)
        self.load_np_data(np_path)
        q_torch, k_torch, v_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            q_torch, k_torch, v_torch, dout_torch
        )
        del q_torch
        del k_torch
        del v_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()
        np.save(
            save_path + '/out.npy',
            self.out_torch
        )
        np.save(
            save_path + '/out_grads.npy', 
            np.array(self.out_grads_torch)
        )

    def save_paddle_res(self, dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax):
        save_path = self.init_paddle_save_path(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)
        np_path = self.init_np_save_path(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)

        self.init_params(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)
        self.load_np_data(np_path)
        q_eager,k_eager,v_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            q_eager,k_eager,v_eager, dout_eager
        )
        del q_eager
        del k_eager
        del v_eager
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        del out_eager
        del out_grads_eager
        np.save(
            save_path + '/out.npy',
            out_eager_np
        )
        np.save(
            save_path + '/out_grads.npy', 
            np.array(out_grads_eager_np)
        )

    def load_np_data(self, np_path):
        self.np_q = np.load(np_path + '/q.npy')
        self.np_k = np.load(np_path + '/k.npy')
        self.np_v = np.load(np_path + '/v.npy')
        self.np_dout = np.load(np_path + '/dout.npy')


    def init_np_save_path(self, dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax):
        args = f"{dtype}_{'_'.join(map(str, shape_q))}_{str(dropout)}_{str(causal)}_{str(return_softmax)}"
        save_path = f'/workspace/PaddleApiTest/test_flash_attention/compare_diff_py_version/np_data/{args}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def init_torch_save_path(self, dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax):
        args = f"{dtype}_{'_'.join(map(str, shape_q))}_{str(dropout)}_{str(causal)}_{str(return_softmax)}"
        save_path = f'/workspace/PaddleApiTest/test_flash_attention/compare_diff_py_version/torch_data/{args}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def init_paddle_save_path(self, dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax):
        args = f"{dtype}_{'_'.join(map(str, shape_q))}_{str(dropout)}_{str(causal)}_{str(return_softmax)}"
        save_path = f'/workspace/PaddleApiTest/test_flash_attention/compare_diff_py_version/paddle_data/{args}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path


    def init_params(self, dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax):
        self.dtype = dtype
        self.shape_q = shape_q
        self.shape_k = shape_k
        self.shape_v = shape_v
        self.dropout = dropout
        self.causal = causal
        self.return_softmax = return_softmax

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_q = np.random.random(size=self.shape_q).astype("float32") - 0.5
        self.np_k = np.random.random(size=self.shape_k).astype("float32") - 0.5
        self.np_v = np.random.random(size=self.shape_v).astype("float32") - 0.5
        self.np_dout = np.random.random(size=self.shape_q).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_q = self.np_q.astype("float16")
            self.np_v = self.np_v.astype("float16")
            self.np_k = self.np_k.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def gen_torch_inputs_and_dout(self):
        q_torch = torch.tensor(
            self.np_q,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        k_torch = torch.tensor(
            self.np_k,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        v_torch = torch.tensor(
            self.np_v,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return q_torch,k_torch,v_torch, dout_torch

    def gen_eager_inputs_and_dout(self):
        q_eager = paddle.to_tensor(
            self.np_q,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        q_eager.stop_gradient = False
        k_eager = paddle.to_tensor(
            self.np_k,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        k_eager.stop_gradient = False
        v_eager = paddle.to_tensor(
            self.np_v,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        v_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return q_eager,k_eager,v_eager, dout_eager



    def cal_torch_res(self, q,k,v, dout):
        q_t = torch.permute(q, (0, 2, 1, 3))
        k_t = torch.permute(k, (0, 2, 1, 3))
        v_t = torch.permute(v, (0, 2, 1, 3))

        if self.dtype == "bfloat16":
            q_t = q_t.to(dtype=torch.bfloat16)
            k_t = k_t.to(dtype=torch.bfloat16)
            v_t = v_t.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)

        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, dropout_p=self.dropout, is_causal=self.causal, scale=None)
        out = torch.permute(out, (0, 2, 1, 3))
        out_grads = torch.autograd.grad([out], [q_t,k_t,v_t], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        out_grads = [out_grad.permute(0, 2, 1, 3) for out_grad in out_grads]
        return out, out_grads

    def cal_eager_res(self, q, k, v, dout):
        if self.dtype == "bfloat16":
            q = paddle.cast(q, dtype="uint16")
            k = paddle.cast(k, dtype="uint16")
            v = paddle.cast(v, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out, _ = flash_attention(q, k, v, dropout=self.dropout, causal=self.causal, return_softmax=self.return_softmax)
        out_grads = paddle.grad([out], [q,k,v], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads



saver = SaveRes()
for i, (shape_q, shape_k, shape_v, dropout, causal, return_softmax) in enumerate(case_list):
    for dtype in ["float16", "bfloat16"]:
        saver.save_paddle_res(dtype, shape_q, shape_k, shape_v, dropout, causal, return_softmax)