import sys
import unittest

import numpy as np
import torch

import paddle

if __name__ == '__main__':
    np.random.seed(2023)
    input = np.random.random(size=[1, 16, 4096, 128])
    torch_output = np.array([1, 16, 4096, 128])
    paddle_output = paddle.shape(paddle.to_tensor(input)).detach().numpy()
    assert((torch_output == paddle_output).all())
    input = np.random.random(size=[1, 8192, 14, 128])
    torch_output = np.array([1, 8192, 14, 128])
    paddle_output = paddle.shape(paddle.to_tensor(input)).detach().numpy()
    assert((torch_output == paddle_output).all())
    print("OK")
