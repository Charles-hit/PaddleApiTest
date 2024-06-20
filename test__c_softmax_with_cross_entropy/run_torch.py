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


def run_torch(ranks, dtype, logits_shape, label_shape):
    dataname = hashlib.md5(
        f"{ranks},{dtype},{logits_shape},{label_shape}".encode("ascii")
    ).digest()
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    logits = []
    for i in range(ranks):
        logits.append(np.load(datadir.joinpath(f"logits{i}.npy")))
    np.concatenate(logits, axis=-1)
    label = np.load(datadir.joinpath(f"labels.npy"))
    logits = torch.tensor(logits, device="cuda", requires_grad=True).transpose(1, 2)
    label = torch.tensor(label, device="cuda").squeeze(-1)
    loss_grad = np.load(datadir.joinpath(f"loss_grad.npy"))
    loss_grad = torch.tensor(loss_grad, device="cuda", requires_grad=True)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(logits, label)
    print(loss)
    print(loss.shape)
    loss.backward(loss_grad)
    print(logits.grad)


def run_para_torch(ranks, dtype, logits_shape, label_shape):
    device_mesh = init_device_mesh("cuda", (8,))
    dataname = hashlib.md5(
        f"{ranks},{dtype},{logits_shape},{label_shape}".encode("ascii")
    ).digest()
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    resultdir = (
        Path(__file__).resolve().parent.joinpath("torch_value").joinpath(dataname)
    )
    if device_mesh.get_rank() == 0:
        os.makedirs(resultdir)

    logits = np.load(datadir.joinpath(f"logits{device_mesh.get_rank()}.npy"))
    logits = np.transpose(logits, (0, 2, 1))
    label = np.load(datadir.joinpath(f"labels.npy"))
    logits = torch.tensor(logits, device="cuda", requires_grad=True)
    label = torch.tensor(label, device="cuda").squeeze(-1)
    loss_grad = np.load(datadir.joinpath(f"loss_grad.npy"))
    loss_grad = torch.tensor(loss_grad, requires_grad=True)

    label = distribute_tensor(label, device_mesh=device_mesh, placements=[Replicate()])
    loss_grad = distribute_tensor(
        loss_grad, device_mesh=device_mesh, placements=[Replicate()]
    )

    dist_input = DTensor.from_local(logits, device_mesh, placements=[Shard(1)])
    with loss_parallel():
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(dist_input, label)
        loss.backward(loss_grad)
        if device_mesh.get_rank() == 0:
            np.save(
                resultdir.joinpath("loss.npy"), loss.to_local().detach().cpu().numpy()
            )
        np.save(
            resultdir.joinpath(f"logits_grad{device_mesh.get_rank()}.npy"),
            logits.grad.detach().transpose(1, 2).cpu().numpy(),
        )
        # print(loss)
        # print(loss.shape)
        # print(logits.grad)


if __name__ == "__main__":
    args = cases[int(sys.argv[1])]
    run_para_torch(*args)
    # run_torch(8, "float32", (1, 8192, 32032), (1, 8192, 1))
