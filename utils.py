# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

TOLERANCE = {
    "float32": {"atol": 1e-6, "rtol": 1e-6},
    "float16": {"atol": 1e-3, "rtol": 1e-3},
    "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
}


def convert_dtype_to_torch_type(dtype):
    import torch

    if dtype == 'float32':
        return torch.float32
    elif dtype == 'float16':
        return torch.float16
    elif dtype == 'bfloat16':
        return torch.bfloat16


def np_assert_accuracy(
    np_a,
    np_b,
    atol,
    rtol,
    dtype,
    version_a,
    version_b,
    eager_or_static_mode,
    fwd_or_bkd,
    api,
):  
    max_atol_idx = np.argmax(np.abs(np_a - np_b))
    np_a_flatten = np_a.flatten()
    np_b_flatten = np_b.flatten()
    sub_res = np_a_flatten - np_b_flatten
    nonzero_idx = np.nonzero(np_b_flatten)
    sub_res = sub_res.take(nonzero_idx)
    np_b_flatten_nonzero = np_b_flatten.take(nonzero_idx).flatten()
    np_a_flatten_nonzero = np_a_flatten.take(nonzero_idx).flatten()
    if sub_res.size ==0:
        max_rtol_idx = 0
    else:
        max_rtol_idx = np.argmax(np.abs(sub_res / np_b_flatten_nonzero))
    np.testing.assert_allclose(
        np_a,
        np_b,
        atol,
        rtol,
        err_msg=(
            '{api} {eager_or_static_mode} {fwd_or_bkd}: compare {version_a} res with {version_b} failed in {dtype} dtype,\n'.format(
                api=api,
                eager_or_static_mode=eager_or_static_mode,
                fwd_or_bkd=fwd_or_bkd,
                version_a=version_a,
                version_b=version_b,
                dtype=dtype,
            )
            + 'max_atol value, {version_a}_value: {value_a}, {version_b}_value: {value_b},\n'.format(
                version_a=version_a,
                value_a=str(np_a_flatten[max_atol_idx].item()),
                version_b=version_b,
                value_b=str(np_b_flatten[max_atol_idx].item()),
            )
            + 'max_rtol value , {version_a}_value: {value_a}, {version_b}_value: {value_b},\n'.format(
                version_a=version_a,
                value_a=str(np_a_flatten_nonzero[max_rtol_idx].item()),
                version_b=version_b,
                value_b=str(np_b_flatten_nonzero[max_rtol_idx].item()),
            )
        ),
    )


def np_assert_staility(
    np_actual,
    np_baseline,
    dtype,
    version,
    eager_or_static_mode,
    fwd_or_bkd,
    api,
):
    max_atol_idx = np.argmax(np.abs(np_actual - np_baseline))
    np_actual_flatten = np_actual.flatten()
    np_baseline_flatten = np_baseline.flatten()
    sub_res = np_actual_flatten - np_baseline_flatten
    nonzero_idx = np.nonzero(np_baseline_flatten)
    sub_res = sub_res.take(nonzero_idx)
    np_baseline_flatten_nonzero = np_baseline_flatten.take(nonzero_idx).flatten()
    if sub_res.size == 0:
        max_rtol_idx = 0
    else:
        np_actual_flatten_nonzero = np_actual_flatten.take(nonzero_idx).flatten()
        max_rtol_idx = np.argmax(np.abs(sub_res / np_baseline_flatten_nonzero))
    np.testing.assert_equal(
        np_actual,
        np_baseline,
        err_msg=(
            '{eager_or_static_mode} {fwd_or_bkd}: {version} is unstable in {dtype} dtype,\n'.format(
                eager_or_static_mode=eager_or_static_mode,
                fwd_or_bkd=fwd_or_bkd,
                version=version,
                dtype=dtype,
            )
            + 'max_atol value, {version}_value: {actual_value}, {version}_baseline_value: {baseline_value}, \n'.format(
                version=version,
                actual_value=str(np_actual_flatten[max_atol_idx].item()),
                baseline_value=str(np_baseline_flatten[max_atol_idx].item()),
            )
            + 'max_rtol value,  {version}_value: {actual_value}, {version}_baseline_value: {baseline_value}, \n'.format(
                version=version,
                actual_value=str(np_actual_flatten_nonzero[max_rtol_idx].item()),
                baseline_value=str(np_baseline_flatten_nonzero[max_rtol_idx].item()),
            )
        ),
    )
