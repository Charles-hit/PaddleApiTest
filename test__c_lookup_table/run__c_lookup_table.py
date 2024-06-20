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
from paddle.distributed.fleet.layers.mpu.mp_ops import _c_lookup_table
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.layers.mpu.mp_layers import VocabParallelEmbedding
import paddle.distributed as dist
from paddle.distributed import collective
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
import paddle.distributed.fleet as fleet
import random

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


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)


def run__c_lookup_table(
    ranks, dtype, table_shape, index_shape, start_index, vacab_size
):
    dataname = hashlib.md5(
        f"{ranks},{dtype},{table_shape},{index_shape},{start_index},{vacab_size}".encode(
            "ascii"
        )
    ).digest()
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    resultdir = (
        Path(__file__).resolve().parent.joinpath("paddle_value").joinpath(dataname)
    )
    if dist.get_rank() == 0:
        resultdir.mkdir(parents=True)

    table = np.load(datadir.joinpath(f"table{dist.get_rank()}.npy"))
    index = np.load(datadir.joinpath(f"index.npy"))
    out_grad = np.load(datadir.joinpath(f"out_grad.npy"))
    index = paddle.to_tensor(index, dtype="int64", stop_gradient=False)
    if dtype == "float32":
        table = paddle.to_tensor(table, dtype="float32", stop_gradient=False)
        out_grad = paddle.to_tensor(out_grad, dtype="float32", stop_gradient=False)
        table = paddle.cast(table, "uint16")
        out_grad = paddle.cast(out_grad, "uint16")
    else:
        table = paddle.to_tensor(table, dtype=dtype, stop_gradient=False)
        out_grad = paddle.to_tensor(out_grad, dtype=dtype, stop_gradient=False)

    group = collective._get_default_group()

    # out = _c_lookup_table(
    #     table,
    #     index,
    #     start_index=dist.get_rank() * table.shape[0],
    #     vocab_size=vacab_size,
    # )
    # out = mp_ops._mp_allreduce(
    #     out,
    #     group=group,
    #     use_calc_stream=True,
    #     use_model_parallel=True,
    # )

    embedding = VocabParallelEmbedding(vacab_size, table.shape[-1], mp_group=group)
    embedding.weight.data = table
    out = embedding(index)

    out.backward(out_grad)
    table_grad = embedding.weight.grad
    # table_grad = table.grad.numpy()

    if dtype == "bfloat16":
        out = paddle.cast(out, "float32")
        table_grad = paddle.cast(table_grad, "float32")

    out = out.numpy()
    table_grad = table_grad.numpy()

    np.save(resultdir.joinpath(f"out.npy"), out)
    np.save(resultdir.joinpath(f"table_grad{dist.get_rank()}.npy"), table_grad)


if __name__ == "__main__":
    dist_strategy = fleet.DistributedStrategy()
    world_size = dist.get_world_size()
    dist_strategy.hybrid_configs = {
        "mp_degree": world_size,
        "pp_degree": 1,
        "dp_degree": 1,
    }
    dist.fleet.init(is_collective=True, strategy=dist_strategy)

    set_random_seed(1024)
    args = cases[int(sys.argv[1])]
    run__c_lookup_table(*args)
