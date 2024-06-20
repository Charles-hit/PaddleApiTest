#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest

import numpy as np
import torch

import paddle
from paddle.utils import map_structure
from paddle.distributed.fleet.layers.mpu.mp_ops import _c_softmax_with_cross_entropy
import paddle.distributed as dist
from paddle.distributed import collective

sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)
import hashlib
from pathlib import Path
from cases import cases


def run__c_softmax_with_cross_entropy(ranks, dtype, logits_shape, label_shape):
    dataname = hashlib.md5(
        f"{ranks},{dtype},{logits_shape},{label_shape}".encode("ascii")
    ).digest()
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    resultdir = (
        Path(__file__).resolve().parent.joinpath("paddle_value").joinpath(dataname)
    )
    if dist.get_rank() == 0:
        resultdir.mkdir(parents=True)

    logits = np.load(datadir.joinpath(f"logits{dist.get_rank()}.npy"))
    label = np.load(datadir.joinpath(f"labels.npy"))
    loss_grad = np.load(datadir.joinpath(f"loss_grad.npy"))
    logits = paddle.to_tensor(logits, dtype=dtype, stop_gradient=False)
    label = paddle.to_tensor(label, dtype="int64", stop_gradient=False)
    loss_grad = paddle.to_tensor(loss_grad, dtype=dtype, stop_gradient=False)
    loss_grad = paddle.unsqueeze(loss_grad, -1)

    group = collective._get_default_group()
    loss = _c_softmax_with_cross_entropy(logits, label, group)
    loss.backward(loss_grad)
    logits_grad = logits.grad.numpy()
    loss = loss.squeeze(-1).numpy()

    np.save(resultdir.joinpath(f"loss.npy"), loss)
    np.save(resultdir.joinpath(f"logits_grad{dist.get_rank()}.npy"), logits_grad)
    # print(loss)
    # print(loss.shape)
    # print(logits_grad)


if __name__ == "__main__":
    dist.init_parallel_env()
    args = cases[int(sys.argv[1])]
    run__c_softmax_with_cross_entropy(*args)
