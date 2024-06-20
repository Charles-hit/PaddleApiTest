import numpy as np
from pathlib import Path
import hashlib
import sys
from cases import cases


def gen_np_inputs(ranks, dtype, table_shape, index_shape, start_index, vacab_size):

    dataname = hashlib.md5(
        f"{ranks},{dtype},{table_shape},{index_shape},{start_index},{vacab_size}".encode(
            "ascii"
        )
    ).digest()
    if dtype == "bfloat16":
        dtype = "float32"
    dataname = dataname.hex()[:6]
    datadir = Path(__file__).resolve().parent.joinpath("data").joinpath(dataname)
    datadir.mkdir(parents=True, exist_ok=True)
    for i in range(ranks):
        table = np.random.random(size=table_shape).astype(dtype)
        np.save(datadir.joinpath(f"table{i}"), table)
    index = np.random.randint(
        low=0, high=table_shape[0] * ranks, size=index_shape
    ).astype("int64")
    np.save(datadir.joinpath(f"index"), index)
    out_grad = np.random.random(size=(*index_shape, table_shape[-1])).astype(dtype)
    np.save(datadir.joinpath(f"out_grad"), out_grad)


if __name__ == "__main__":
    np.random.seed(2024)
    args = cases[int(sys.argv[1])]
    gen_np_inputs(*args)
