import torch
import hashlib
from pathlib import Path
import numpy as np
from torch.distributed.tensor.parallel import loss_parallel
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, distribute_tensor, Replicate, DTensor
import os
import sys
from cases import cases
from torch import nn


def run_torch(ranks, dtype, table_shape, index_shape, start_index, vacab_size):
    dataname = hashlib.md5(
        f"{ranks},{dtype},{table_shape},{index_shape},{start_index},{vacab_size}".encode(
            "ascii"
        )
    ).digest()
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    resultdir = (
        Path(__file__).resolve().parent.joinpath("torch_value").joinpath(dataname)
    )
    os.makedirs(resultdir)

    table = []
    for i in range(ranks):
        table.append(np.load(datadir.joinpath(f"table{i}.npy")))
    table = np.concatenate(table, axis=0)
    index = np.load(datadir.joinpath(f"index.npy"))

    table = torch.tensor(table, device="cuda", requires_grad=True)
    index = torch.tensor(index, device="cuda")
    out_grad = np.load(datadir.joinpath(f"out_grad.npy"))
    out_grad = torch.tensor(out_grad, device="cuda", requires_grad=True)
    if dtype == "bfloat16":
        table = table.bfloat16()
        out_grad = out_grad.bfloat16()

    embedding = nn.Embedding(vacab_size, table_shape[1])
    embedding.weight = torch.nn.Parameter(table, requires_grad=True)
    embedding.zero_grad()
    out = embedding(index)
    out.backward(out_grad)

    out = out.detach()
    table_grad = embedding.weight.grad.detach()
    if dtype == "bfloat16":
        out = out.cpu().to(torch.float32)
        table_grad = table_grad.cpu().to(torch.float32)

    np.save(resultdir.joinpath(f"out.npy"), out.numpy())
    np.save(
        resultdir.joinpath(f"table_grad.npy"),
        table_grad.numpy(),
    )


if __name__ == "__main__":
    args = cases[int(sys.argv[1])]
    run_torch(*args)
