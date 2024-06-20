import numpy as np
from pathlib import Path
import hashlib
import sys
from cases import cases


def gen_np_inputs(ranks, dtype, logits_shape, label_shape):
    dataname = hashlib.md5(
        f"{ranks},{dtype},{logits_shape},{label_shape}".encode("ascii")
    ).digest()
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    datadir.mkdir(parents=True, exist_ok=True)
    for i in range(ranks):
        logits = np.random.random(size=logits_shape).astype(dtype)
        np.save(datadir.joinpath(f"logits{i}"), logits)
    label = np.random.randint(
        low=0, high=logits_shape[-1] * ranks, size=label_shape
    ).astype("int64")
    np.save(datadir.joinpath(f"labels"), label)
    loss_grad = np.random.random(size=label_shape[:-1]).astype(dtype)
    np.save(datadir.joinpath(f"loss_grad"), loss_grad)


if __name__ == "__main__":
    np.random.seed(2024)
    args = cases[int(sys.argv[1])]
    gen_np_inputs(*args)
