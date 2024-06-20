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
import multiprocess


def check(ranks, dtype, table_shape, index_shape, start_index, vacab_size):
    atol = TOLERANCE[dtype]["atol"]
    rtol = TOLERANCE[dtype]["rtol"]
    dataname = hashlib.md5(
        f"{ranks},{dtype},{table_shape},{index_shape},{start_index},{vacab_size}".encode(
            "ascii"
        )
    ).digest()
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    paddle_dir = (
        Path(__file__).resolve().parent.joinpath("paddle_value").joinpath(dataname)
    )
    torch_dir = (
        Path(__file__).resolve().parent.joinpath("torch_value").joinpath(dataname)
    )

    out_paddle = np.load(paddle_dir.joinpath(f"out.npy"))
    out_torch = np.load(torch_dir.joinpath(f"out.npy"))
    np_assert_accuracy(
        out_paddle,
        out_torch,
        atol,
        rtol,
        dtype,
        version_a="paddle_develop",
        version_b="torch",
        eager_or_static_mode="eager",
        fwd_or_bkd="forward",
        api="_c_lookup_table",
    )
    table_grad_paddle = []

    for i in range(0, ranks):
        table_grad_i_paddle = np.load(paddle_dir.joinpath(f"table_grad{i}.npy"))
        table_grad_paddle.append(table_grad_i_paddle)
    table_grad_paddle = np.concatenate(table_grad_paddle, axis=0)
    table_grad_torch = np.load(torch_dir.joinpath(f"table_grad.npy"))
    np_assert_accuracy(
        table_grad_paddle,
        table_grad_torch,
        atol,
        rtol,
        dtype,
        version_a="paddle_develop",
        version_b="torch",
        eager_or_static_mode="eager",
        fwd_or_bkd="backward",
        api="_c_lookup_table",
    )
    print("OK")


if __name__ == "__main__":
    args = cases[int(sys.argv[1])]
    check(*args)
