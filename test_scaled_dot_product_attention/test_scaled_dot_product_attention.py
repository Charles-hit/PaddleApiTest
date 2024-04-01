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

# There is no FP32 kernel for GPU
class TestScaledDotProductAttentionDevelopCase1_FP16(unittest.TestCase):
    def init_dtype(self):
        self.dtype = "float16"

    def init_params(self):
        self.q_shape = [1, 1, 2, 40]
        self.k_shape = [1, 1, 2, 40]
        self.v_shape = [1, 1, 2, 40]
        self.dout_shape = [1, 1, 2, 40]
        self.attn_mask_shape = None
        self.dropout_p = 0.0
        self.is_causal = False
        self.training = True
        self.init_dtype()

    def setUp(self):
        paddle.seed(1127)
        torch.manual_seed(1127)
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        q_torch, k_torch, v_torch, attn_mask_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            q_torch, k_torch, v_torch, attn_mask_torch, self.dropout_p, self.is_causal, self.training, dout_torch
        )
        del q_torch
        del k_torch
        del v_torch
        if self.attn_mask_shape:
            del attn_mask_torch
        del dout_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        def to_numpy(x):
            if self.dtype == "bfloat16":
                return x.cpu().to(dtype=torch.float32).numpy()
            return x.cpu().numpy()
        self.out_grads_torch = map_structure(
            to_numpy,
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_np_inputs_and_dout(self):
        self.np_q = np.random.random(size=self.q_shape).astype("float32") - 0.5
        self.np_k = np.random.random(size=self.k_shape).astype("float32") - 0.5
        self.np_v = np.random.random(size=self.v_shape).astype("float32") - 0.5
        if self.attn_mask_shape is not None:
            self.np_attn_mask = np.random.random(size=self.attn_mask_shape).astype("float32") - 0.5
        else:
            self.np_attn_mask = None
        self.np_dout = (
            np.random.random(size=self.dout_shape).astype("float32") - 0.5
        )
        # convert np array dtype
        if self.dtype == "float16":
            self.np_q = self.np_q.astype("float16")
            self.np_k = self.np_k.astype("float16")
            self.np_v = self.np_v.astype("float16")
            if self.attn_mask_shape is not None:
                self.np_attn_mask = self.np_attn_mask.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

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
        if self.attn_mask_shape is not None:
            attn_mask_torch = torch.tensor(
                self.np_attn_mask,
                device='cuda',
                dtype=convert_dtype_to_torch_type(self.dtype)
                if self.dtype != 'bfloat16'
                else torch.float32,
                requires_grad=True,
            )
        else:
            attn_mask_torch = None
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return q_torch, k_torch, v_torch, attn_mask_torch, dout_torch

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
        if self.attn_mask_shape is not None:
            attn_mask_eager = paddle.to_tensor(
                self.np_attn_mask,
                dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
                place="gpu",
            )
            attn_mask_eager.stop_gradient = False
        else:
            attn_mask_eager = None
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return q_eager, k_eager, v_eager, attn_mask_eager, dout_eager

    def cal_torch_res(self, q, k, v, attn_mask, dropout_p, is_causal, training, dout):
        q_t = q
        k_t = k
        v_t = v
        attn_mask_t = attn_mask
        dout_t = dout
        if self.dtype == "bfloat16":
            q_t = q.to(dtype=torch.bfloat16)
            k_t = k.to(dtype=torch.bfloat16)
            v_t = v.to(dtype=torch.bfloat16)
            if self.attn_mask_shape is not None:
                attn_mask_t = attn_mask.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        q_t = q_t.permute(0, 2, 1, 3)
        k_t = k_t.permute(0, 2, 1, 3)
        v_t = v_t.permute(0, 2, 1, 3)
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask_t, dropout_p=dropout_p, is_causal=is_causal)
        out = out.permute(0, 2, 1, 3)
        out_grads = torch.autograd.grad([out], [q_t, k_t, v_t], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        out_grads = [out_grad.permute(0, 2, 1, 3) for out_grad in out_grads]
        return out, out_grads

    def cal_eager_res(self, q, k, v, attn_mask, dropout_p, is_causal, training, dout):
        q_t = q
        k_t = k
        v_t = v
        attn_mask_t = attn_mask
        dout_t = dout
        if self.dtype == "bfloat16":
            q_t = paddle.cast(q, dtype="uint16")
            k_t = paddle.cast(k, dtype="uint16")
            v_t = paddle.cast(v, dtype="uint16")
            if self.attn_mask_shape is not None:
                attn_mask_t = paddle.cast(attn_mask, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        assert training is True, "In pytorch, training always be True"
        # with paddle.nn.functional.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        out = paddle.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask_t, dropout_p=dropout_p, is_causal=is_causal, training=training)
        # breakpoint()
        out_grads = paddle.grad([out], [q, k, v], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        q_eager, k_eager, v_eager, attn_mask_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            q_eager, k_eager, v_eager, attn_mask_eager, self.dropout_p, self.is_causal, self.training, dout_eager
        )
        del q_eager
        del k_eager
        del v_eager
        if self.attn_mask_shape is not None:
            del attn_mask_eager
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        del out_eager
        del out_grads_eager
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
            api="paddle.nn.functional.scaled_dot_product_attention",
        )
        # compare develop eager backward res with torch
        for idx in range(len(out_grads_eager_np)):
            np_assert_accuracy(
                out_grads_eager_np[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="backward",
                api="paddle.nn.functional.scaled_dot_product_attention",
            )


class TestScaledDotProductAttentionDevelopCase2_FP16(TestScaledDotProductAttentionDevelopCase1_FP16):
    def init_params(self):
        self.q_shape = [10, 1, 16, 96]
        self.k_shape = [10, 1, 16, 96]
        self.v_shape = [10, 1, 16, 96]
        self.dout_shape = [10, 1, 16, 96]
        self.attn_mask_shape = None
        self.dropout_p = 0.0
        self.is_causal = False
        self.training = True
        self.init_dtype()

class TestScaledDotProductAttentionDevelopCase2_BF16(TestScaledDotProductAttentionDevelopCase2_FP16):
    def init_dtype(self):
        self.dtype = "bfloat16"

class TestScaledDotProductAttentionDevelopCase3_FP16(TestScaledDotProductAttentionDevelopCase1_FP16):
    def init_params(self):
        self.q_shape = [10, 181, 16, 96]
        self.k_shape = [10, 181, 16, 96]
        self.v_shape = [10, 181, 16, 96]
        self.dout_shape = [10, 181, 16, 96]
        self.attn_mask_shape = None
        self.dropout_p = 0.0
        self.is_causal = False
        self.training = True
        self.init_dtype()

class TestScaledDotProductAttentionDevelopCase3_BF16(TestScaledDotProductAttentionDevelopCase3_FP16):
    def init_dtype(self):
        self.dtype = "bfloat16"

class TestScaledDotProductAttentionDevelopCase4_FP16(TestScaledDotProductAttentionDevelopCase1_FP16):
    def init_params(self):
        self.q_shape = [1, 2880, 1, 512]
        self.k_shape = [1, 2880, 1, 512]
        self.v_shape = [1, 2880, 1, 512]
        self.dout_shape = [1, 2880, 1, 512]
        self.attn_mask_shape = None
        self.dropout_p = 0.0
        self.is_causal = False
        self.training = True
        self.init_dtype()

class TestScaledDotProductAttentionDevelopCase4_BF16(TestScaledDotProductAttentionDevelopCase4_FP16):
    def init_dtype(self):
        self.dtype = "bfloat16"

class TestScaledDotProductAttentionDevelopCase5_FP16(TestScaledDotProductAttentionDevelopCase1_FP16):
    def init_params(self):
        self.q_shape = [10, 180, 16, 96]
        self.k_shape = [10, 180, 16, 96]
        self.v_shape = [10, 180, 16, 96]
        self.dout_shape = [10, 180, 16, 96]
        self.attn_mask_shape = None
        self.dropout_p = 0.0
        self.is_causal = False
        self.training = True
        self.init_dtype()

class TestScaledDotProductAttentionDevelopCase5_BF16(TestScaledDotProductAttentionDevelopCase5_FP16):
    def init_dtype(self):
        self.dtype = "bfloat16"

class TestScaledDotProductAttentionDevelopCase6_FP16(TestScaledDotProductAttentionDevelopCase1_FP16):
    def init_params(self):
        self.q_shape = [1, 1, 2, 40]
        self.k_shape = [1, 1, 2, 40]
        self.v_shape = [1, 1, 2, 40]
        self.dout_shape = [1, 1, 2, 40]
        self.attn_mask_shape = [1, 2, 1, 1]
        self.dropout_p = 0.0
        self.is_causal = False
        self.training = True
        self.init_dtype()

class TestScaledDotProductAttentionDevelopCase6_BF16(TestScaledDotProductAttentionDevelopCase6_FP16):
    def init_dtype(self):
        self.dtype = "bfloat16"


if __name__ == '__main__':
    unittest.main()