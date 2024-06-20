import sys
sys.path.append("..")
import numpy as np
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)
import hashlib
from pathlib import Path
from cases import cases


def check(ranks, dtype, logits_shape, label_shape):
    atol = TOLERANCE[dtype]["atol"]
    rtol = TOLERANCE[dtype]["rtol"]
    dataname = hashlib.md5(
        f"{ranks},{dtype},{logits_shape},{label_shape}".encode("ascii")
    ).digest()
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    paddle_dir = (
        Path(__file__).resolve().parent.joinpath("paddle_value").joinpath(dataname)
    )
    torch_dir = (
        Path(__file__).resolve().parent.joinpath("torch_value").joinpath(dataname)
    )
    loss_paddle = np.load(paddle_dir.joinpath(f"loss.npy"))
    loss_torch = np.load(torch_dir.joinpath(f"loss.npy"))
    np_assert_accuracy(
        loss_paddle,
        loss_torch,
        atol,
        rtol,
        dtype,
        version_a="paddle_develop",
        version_b="torch",
        eager_or_static_mode="eager",
        fwd_or_bkd="forward",
        api="_c_softmax_with_cross_entropy",
    )
    for i in range(0, ranks):
        logits_grad_paddle = np.load(paddle_dir.joinpath(f"logits_grad{i}.npy"))
        logits_grad_torch = np.load(torch_dir.joinpath(f"logits_grad{i}.npy"))
        np_assert_accuracy(
            logits_grad_paddle,
            logits_grad_torch,
            atol,
            rtol,
            dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="eager",
            fwd_or_bkd="backward",
            api="_c_softmax_with_cross_entropy",
        )
    print("OK")


if __name__ == "__main__":
    args = cases[int(sys.argv[1])]
    check(*args)
