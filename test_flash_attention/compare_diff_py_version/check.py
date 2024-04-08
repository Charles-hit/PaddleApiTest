import os
import sys

import numpy as np
from paddle import dtype

sys.path.append("...")
from utils import (
    TOLERANCE,
)


paddle_path = '/workspace/PaddleApiTest/test_flash_attention/compare_diff_py_version/paddle_data'
torch_path = '/workspace/PaddleApiTest/test_flash_attention/compare_diff_py_version/torch_data'

for sub_dir in os.listdir(paddle_path):
    dtype = sub_dir.split('_')[0]
    atol = TOLERANCE[dtype]["atol"]
    rtol = TOLERANCE[dtype]["rtol"]
    for file in os.listdir(os.path.join(paddle_path, sub_dir)):
        paddle_data = np.load(os.path.join(paddle_path, sub_dir, file))
        torch_data = np.load(os.path.join(torch_path, sub_dir, file))
        # print(paddle_data)
        # print(torch_data)
        np.testing.assert_allclose(paddle_data, torch_data, rtol=rtol, atol=atol)
        print(f'finish check {sub_dir}/{file}.')