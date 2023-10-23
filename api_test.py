import gc
import random
import unittest
from typing import Sequence

import numpy as np
import torch
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

import paddle
from paddle.utils import map_structure


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, Sequence) else [x]


def flatten(nest_list: Sequence):
    out = []
    for item in nest_list:
        if isinstance(item, Sequence):
            tmp_list = flatten(item)
            out += tmp_list
        else:
            out.append(item)
    return out


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(2023)
        random.seed(2023)
        paddle.seed(2023)
        torch.manual_seed(2023)
        torch.cuda.manual_seed_all(2023)

    @classmethod
    def tearDownClass(cls):
        pass

    def check_inputs_and_out_grads(self):
        if not hasattr(self, "inputs"):
            raise TypeError(
                "Please set self.inputs to a list of numpy arrays in setUp function."
            )
        if not isinstance(self.inputs, list):
            raise TypeError(
                "Please must set self.inputs to a list of numpy arrays in setUp function."
            )
        if hasattr(self, "out_grads") and not isinstance(self.out_grads, list):
            raise TypeError(
                "You must set self.out_grads to a list of numpy arrays in setUp function."
            )

    def check_dtype(self):
        if not hasattr(self, "dtype"):
            raise TypeError("You must set self.dtype in setUp function.")
        expected_dtype = ["float32", "float16", "bfloat16"]
        if self.dtype not in expected_dtype:
            raise TypeError(
                f"The data type must be {expected_dtype}, but received {self.dtype}."
            )

    def gen_eager_data(self, np_xs, dtype):
        eager_xs = []
        if isinstance(np_xs, Sequence):
            for np_x in np_xs:
                if isinstance(np_x, Sequence):
                    eager_xs.append(self.gen_eager_data(np_x, dtype))
                else:
                    if dtype == "bfloat16" and np_x.dtype == np.float32:
                        x = paddle.to_tensor(
                            np_x,
                            dtype="float32",
                            place="gpu",
                            stop_gradient=False,
                        )
                        x = paddle.cast(x, dtype="uint16")
                    else:
                        x = paddle.to_tensor(
                            np_x,
                            dtype=np_x.dtype,
                            place="gpu",
                            stop_gradient=False,
                        )
                    eager_xs.append(x)
        else:
            if dtype == "bfloat16" and np_xs.dtype == np.float32:
                x = paddle.to_tensor(
                    np_xs, dtype="float32", place="gpu", stop_gradient=False
                )
                x = paddle.cast(x, dtype=dtype)
            else:
                x = paddle.to_tensor(
                    np_xs, dtype=np_xs.dtype, place="gpu", stop_gradient=False
                )
            eager_xs.append(x)
        return eager_xs

    def gen_torch_data(self, np_xs, dtype):
        eager_xs = []
        if isinstance(np_xs, Sequence):
            for np_x in np_xs:
                if isinstance(np_x, Sequence):
                    eager_xs.append(self.gen_eager_data(np_x, dtype))
                else:
                    if dtype == "bfloat16" and np_x.dtype == np.float32:
                        x = torch.tensor(
                            np_x,
                            device='cuda',
                            dtype=torch.float32,
                            requires_grad=True,
                        )
                        x = x.to(dtype=torch.bfloat16)
                    else:
                        x = torch.tensor(
                            np_x,
                            device='cuda',
                            dtype=convert_dtype_to_torch_type(np_x.dtype),
                            requires_grad=True,
                        )
                    eager_xs.append(x)
        else:
            if dtype == "bfloat16" and np_xs.dtype == np.float32:
                x = torch.tensor(
                    np_xs,
                    device='cuda',
                    dtype=torch.float32,
                    requires_grad=True,
                )
                x = x.to(dtype=torch.bfloat16)
            else:
                x = torch.tensor(
                    np_xs,
                    device='cuda',
                    dtype=convert_dtype_to_torch_type(np_xs.dtype),
                    requires_grad=True,
                )
            eager_xs.append(x)
        return eager_xs

    def gen_static_data_and_feed(self, np_xs, dtype, base_name):
        feed = {}
        static_xs = []
        if isinstance(np_xs, Sequence):
            for i, x in enumerate(np_xs):
                if isinstance(x, Sequence):
                    xs_sub, feed_sub = self.gen_static_data_and_feed(
                        x, f"{base_name}_{i}"
                    )
                    static_xs.append(xs_sub)
                    feed.update(feed_sub)
                else:
                    if dtype == "bfloat16" and x.dtype == np.float32:
                        data = paddle.static.data(
                            f"{base_name}_{i}", x.shape, "float32"
                        )
                        data = paddle.cast(data, dtype="uint16")
                    else:
                        data = paddle.static.data(
                            f"{base_name}_{i}", x.shape, x.dtype
                        )
                    data.stop_gradient = False
                    static_xs.append(data)
                    feed.update({f"{base_name}_{i}": x})
        else:
            if dtype == "bfloat16" and x.dtype == np.float32:
                data = paddle.static.data(
                    f"{base_name}_{i}", np_xs.shape, "float32"
                )
                data = paddle.cast(data, dtype="uint16")
            else:
                data = paddle.static.data(
                    f"{base_name}_{i}", np_xs.shape, np_xs.dtype
                )
            data.stop_gradient = False
            static_xs.append(data)
            feed.update({f"{base_name}_{i}": np_xs})
        return static_xs, feed

    def check_custom_config(self):
        self.check_inputs_and_out_grads()
        self.check_dtype()

    def get_default_threshold(self):
        return TOLERANCE[self.dtype]

    def cal_torch_res(self, inputs, out_grads=None):
        raise NotImplementedError(
            "You must implement cal_torch_res function in your test case."
        )

    def cal_paddle_res(self, inputs, out_grads=None):
        raise NotImplementedError(
            "You must implement cal_paddle_res function in your test case."
        )

    def check_eager_res(self, atol=None, rtol=None):
        self.check_custom_config()
        default_threshold_mp = self.get_default_threshold()
        atol = atol if atol else default_threshold_mp["atol"]
        rtol = rtol if rtol else default_threshold_mp["rtol"]

        torch_inputs = self.gen_torch_data(self.inputs, dtype=self.dtype)
        torch_douts = (
            self.gen_torch_data(self.out_grads, dtype=self.dtype)
            if hasattr(self, "out_grads")
            else None
        )
        torch_outputs, torch_gradouts = self.cal_torch_res(
            torch_inputs, torch_douts
        )
        if torch_gradouts is None:
            torch_gradouts = []
        torch_outputs, torch_gradouts = flatten(
            _as_list(torch_outputs)
        ), flatten(_as_list(torch_gradouts))
        if self.dtype == "bfloat16":
            torch_outputs = (
                map_structure(lambda x: x.to(torch.float32), torch_outputs)
                if len(torch_outputs) > 0
                else torch_outputs
            )
            torch_gradouts = (
                map_structure(lambda x: x.to(torch.float32), torch_gradouts)
                if len(torch_gradouts) > 0
                else torch_gradouts
            )
        torch_outputs_np = map_structure(
            lambda x: x.cpu().detach().numpy(),
            torch_outputs,
        )
        torch_gradouts_np = map_structure(
            lambda x: x.cpu().detach().numpy(),
            torch_gradouts,
        )
        del torch_inputs
        del torch_douts
        del torch_outputs
        del torch_gradouts
        gc.collect()
        torch.cuda.empty_cache()
        pd_inputs = self.gen_eager_data(self.inputs, dtype=self.dtype)
        pd_douts = (
            self.gen_eager_data(self.out_grads, dtype=self.dtype)
            if hasattr(self, "out_grads")
            else None
        )
        pd_outputs, pd_gradouts = self.cal_paddle_res(pd_inputs, pd_douts)
        if pd_douts is None:
            pd_gradouts = []
        pd_outputs, pd_gradouts = flatten(_as_list(pd_outputs)), flatten(
            _as_list(pd_gradouts)
        )
        if self.dtype == "bfloat16":
            pd_outputs = (
                map_structure(lambda x: paddle.cast(x, "float32"), pd_outputs)
                if len(pd_outputs) > 0
                else pd_outputs
            )
            pd_gradouts = (
                map_structure(lambda x: paddle.cast(x, "float32"), pd_gradouts)
                if len(pd_gradouts) > 0
                else pd_gradouts
            )
        pd_outputs_np = map_structure(
            lambda x: x.numpy(),
            pd_outputs,
        )
        pd_gradouts_np = map_structure(
            lambda x: x.numpy(),
            pd_gradouts,
        )
        del pd_inputs
        del pd_douts
        del pd_outputs
        del pd_gradouts
        gc.collect()
        paddle.device.cuda.empty_cache()

        np.testing.assert_equal(
            len(pd_outputs_np),
            len(torch_outputs_np),
            err_msg=(
                'Mismatch between paddle and torch forward output tensor nums.'
                'paddle output tensor num: {}, torch output tensor num: {}.\n'.format(str(len(pd_outputs_np)), str(len(torch_outputs_np)))
            ),
        )

        for idx in range(len(torch_outputs_np)):
            np_assert_accuracy(
                pd_outputs_np[idx],
                torch_outputs_np[idx],
                atol,
                rtol,
                self.dtype,
                version_a="paddle",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="",
            )
        np.testing.assert_equal(
            len(pd_gradouts_np),
            len(torch_gradouts_np),
            err_msg=(
                'Mismatch between paddle and torch grad output tensor nums in eager mode.'
                'paddle grad output tensor num: {}, torch grad output tensor num: {}.\n'.format(str(len(pd_gradouts_np)), str(len(torch_gradouts_np)))
            ),
        )

        for idx in range(len(torch_gradouts_np)):
            np_assert_accuracy(
                pd_gradouts_np[idx],
                torch_gradouts_np[idx],
                atol,
                rtol,
                self.dtype,
                version_a="paddle",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="grad",
                api="",
            )

    def check_static_res(self, atol=None, rtol=None):
        self.check_custom_config()
        default_threshold_mp = self.get_default_threshold()
        atol = atol if atol else default_threshold_mp["atol"]
        rtol = rtol if rtol else default_threshold_mp["rtol"]

        torch_inputs = self.gen_torch_data(self.inputs, dtype=self.dtype)
        torch_douts = (
            self.gen_torch_data(self.out_grads, dtype=self.dtype)
            if hasattr(self, "out_grads")
            else None
        )
        torch_outputs, torch_gradouts = self.cal_torch_res(
            torch_inputs, torch_douts
        )
        if torch_gradouts is None:
            torch_gradouts = []
        torch_outputs, torch_gradouts = flatten(
            _as_list(torch_outputs)
        ), flatten(_as_list(torch_gradouts))
        if self.dtype == "bfloat16":
            torch_outputs = (
                map_structure(lambda x: x.to(torch.float32), torch_outputs)
                if len(torch_outputs) > 0
                else torch_outputs
            )
            torch_gradouts = (
                map_structure(lambda x: x.to(torch.float32), torch_gradouts)
                if len(torch_gradouts) > 0
                else torch_gradouts
            )
        torch_outputs_np = map_structure(
            lambda x: x.cpu().detach().numpy(),
            torch_outputs,
        )
        torch_gradouts_np = map_structure(
            lambda x: x.cpu().detach().numpy(),
            torch_gradouts,
        )
        del torch_inputs
        del torch_douts
        del torch_outputs
        del torch_gradouts
        gc.collect()
        torch.cuda.empty_cache()
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                static_inputs, inputs_feed = self.gen_static_data_and_feed(
                    self.inputs, dtype=self.dtype, base_name="x"
                )
                (static_douts, douts_feed) = (
                    self.gen_static_data_and_feed(
                        self.out_grads, dtype=self.dtype, base_name="dout"
                    )
                    if hasattr(self, "out_grads")
                    else (None, {})
                )
                pd_outputs, pd_gradouts = self.cal_paddle_res(
                    static_inputs, static_douts
                )
                if pd_gradouts is None:
                    pd_gradouts = []
                pd_outputs, pd_gradouts = flatten(
                    _as_list(pd_outputs)
                ), flatten(_as_list(pd_gradouts))
                if self.dtype == "bfloat16":
                    pd_outputs = (
                        map_structure(
                            lambda x: paddle.cast(x, "float32"), pd_outputs
                        )
                        if len(pd_outputs) > 0
                        else pd_outputs
                    )
                    pd_gradouts = (
                        map_structure(
                            lambda x: paddle.cast(x, "float32"), pd_gradouts
                        )
                        if len(pd_gradouts) > 0
                        else pd_gradouts
                    )
                feed = {**inputs_feed, **douts_feed}
                exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
                exe.run(sp)
                out = exe.run(
                    mp,
                    feed=feed,
                    fetch_list=pd_outputs + pd_gradouts,
                )
        if len(pd_gradouts) > 0:
            out_static, out_grads_static = (
                out[0 : len(pd_outputs)],
                out[len(pd_outputs) :],
            )
        else:
            out_static, out_grads_static = out, []
        np.testing.assert_equal(
            len(out_static),
            len(torch_outputs_np),
            err_msg=(
                'Mismatch between paddle static and torch forward output tensor nums.'
                'paddle static mode output tensor num: {}, torch output tensor num: {}.\n'.format(str(len(out_static)), str(len(torch_outputs_np)))
            ),
        )

        for idx in range(len(torch_outputs_np)):
            np_assert_accuracy(
                out_static[idx],
                torch_outputs_np[idx],
                atol,
                rtol,
                self.dtype,
                version_a="paddle",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="forward",
                api="",
            )
        np.testing.assert_equal(
            len(out_grads_static),
            len(torch_gradouts_np),
            err_msg=(
                'Mismatch between paddle  and torch grad output tensor nums in staitc mode.'
                'paddle grad output tensor num: {}, torch grad output tensor num: {}.\n'.format(str(len(out_grads_static)), str(len(torch_gradouts_np)))
            ),
        )

        for idx in range(len(torch_gradouts_np)):
            np_assert_accuracy(
                out_grads_static[idx],
                torch_gradouts_np[idx],
                atol,
                rtol,
                self.dtype,
                version_a="paddle",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="grad",
                api="",
            )

    def check_eager_stability(self, frequency=5):
        self.check_custom_config()
        x_eager = self.gen_eager_data(self.inputs, dtype=self.dtype)
        dout_eager = (
            self.gen_eager_data(self.out_grads, dtype=self.dtype)
            if hasattr(self, "out_grads")
            else None
        )
        out_eager_baseline, out_grads_eager_baseline = self.cal_paddle_res(
            x_eager, dout_eager
        )
        if out_grads_eager_baseline is None:
            out_grads_eager_baseline = []
        out_eager_baseline, out_grads_eager_baseline = flatten(
            _as_list(out_eager_baseline)
        ), flatten(_as_list(out_grads_eager_baseline))
        if self.dtype == "bfloat16":
            out_eager_baseline = (
                map_structure(
                    lambda x: paddle.cast(x, "float32"), out_eager_baseline
                )
                if len(out_eager_baseline) > 0
                else out_eager_baseline
            )
            out_grads_eager_baseline = (
                map_structure(
                    lambda x: paddle.cast(x, "float32"),
                    out_grads_eager_baseline,
                )
                if len(out_grads_eager_baseline) > 0
                else out_grads_eager_baseline
            )
        out_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_eager_baseline,
        )
        out_grads_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager_baseline,
        )
        del out_eager_baseline
        del out_grads_eager_baseline
        gc.collect()
        paddle.device.cuda.empty_cache()

        for i in range(frequency):
            out_eager, out_grads_eager = self.cal_paddle_res(
                x_eager, dout_eager
            )
            if out_grads_eager is None:
                out_grads_eager = []
            out_eager, out_grads_eager = flatten(_as_list(out_eager)), flatten(
                _as_list(out_grads_eager)
            )
            if self.dtype == "bfloat16":
                out_eager = (
                    map_structure(
                        lambda x: paddle.cast(x, "float32"), out_eager
                    )
                    if len(out_eager) > 0
                    else out_eager
                )
                out_grads_eager = (
                    map_structure(
                        lambda x: paddle.cast(x, "float32"), out_grads_eager
                    )
                    if len(out_grads_eager) > 0
                    else out_grads_eager
                )
            out_eager_np = map_structure(
                lambda x: x.numpy(),
                out_eager,
            )
            out_grads_eager_np = map_structure(
                lambda x: x.numpy(),
                out_grads_eager,
            )

            for idx in range(len(out_eager_baseline_np)):
                np_assert_staility(
                    out_eager_np[idx],
                    out_eager_baseline_np[idx],
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="",
                )

            for idx in range(len(out_grads_eager_baseline_np)):
                np_assert_staility(
                    out_grads_eager_np[idx],
                    out_grads_eager_baseline_np[idx],
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="",
                )

    def check_static_stability(self, frequency=5):
        self.check_custom_config()

        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                static_inputs, inputs_feed = self.gen_static_data_and_feed(
                    self.inputs, dtype=self.dtype, base_name="x"
                )
                static_douts, douts_feed = (
                    self.gen_static_data_and_feed(
                        self.out_grads, dtype=self.dtype, base_name="dout"
                    )
                    if hasattr(self, "out_grads")
                    else (None, {})
                )
                (pd_outputs, pd_gradouts) = self.cal_paddle_res(
                    static_inputs, static_douts
                )
                if pd_gradouts is None:
                    pd_gradouts = []
                pd_outputs, pd_gradouts = flatten(
                    _as_list(pd_outputs)
                ), flatten(_as_list(pd_gradouts))
                if self.dtype == "bfloat16":
                    pd_outputs = (
                        map_structure(
                            lambda x: paddle.cast(x, "float32"), pd_outputs
                        )
                        if len(pd_outputs) > 0
                        else pd_outputs
                    )
                    pd_gradouts = (
                        map_structure(
                            lambda x: paddle.cast(x, "float32"), pd_gradouts
                        )
                        if len(pd_gradouts) > 0
                        else pd_gradouts
                    )
                feed = {**inputs_feed, **douts_feed}
                exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
                exe.run(sp)
                out = exe.run(
                    mp,
                    feed=feed,
                    fetch_list=pd_outputs + pd_gradouts,
                )
            if len(pd_gradouts) > 0:
                out_static_baseline, out_grads_static_baseline = (
                    out[0 : len(pd_outputs)],
                    out[len(pd_outputs) :],
                )
            else:
                out_static_baseline, out_grads_static_baseline = out, []
            for i in range(frequency):
                out = exe.run(
                    mp,
                    feed=feed,
                    fetch_list=pd_outputs + pd_gradouts,
                )
                if len(pd_gradouts) > 0:
                    out_static, out_grads_static = (
                        out[0 : len(pd_outputs)],
                        out[len(pd_outputs) :],
                    )
                else:
                    out_static, out_grads_static = out, []
                # test develop static forward stability
                for idx in range(len(out_static_baseline)):
                    np_assert_staility(
                        out_static[idx],
                        out_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="forward",
                        api="",
                    )
                # test develop static backward stability
                for idx in range(len(out_grads_static_baseline)):
                    np_assert_staility(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="backward",
                        api="",
                    )
