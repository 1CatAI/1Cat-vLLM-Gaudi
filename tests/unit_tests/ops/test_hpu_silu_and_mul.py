# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import torch
import torch.nn.functional as F

from vllm_gaudi.ops.hpu_silu_and_mul import HPUSiluAndMul


def test_hpu_silu_and_mul_uses_native_path_by_default():
    x = torch.randn(8, 32)
    with mock.patch.dict("os.environ", {}, clear=True):
        output = HPUSiluAndMul.forward_oot(x)

    gate, up = x.chunk(2, dim=-1)
    torch.testing.assert_close(output, F.silu(gate) * up)


def test_hpu_silu_and_mul_uses_explicit_sigmoid_for_long_prompts():
    x = torch.randn(8, 32)
    env = {
        "VLLM_HPU_EXPLICIT_SIGMOID_SILU": "true",
        "VLLM_HPU_EXPLICIT_SIGMOID_SILU_MIN_TOKENS": "8",
    }
    with mock.patch.dict("os.environ", env, clear=True):
        output = HPUSiluAndMul.forward_oot(x)

    gate, up = x.chunk(2, dim=-1)
    torch.testing.assert_close(output, (gate * torch.sigmoid(gate)) * up)


def test_hpu_silu_and_mul_keeps_native_path_below_threshold():
    x = torch.randn(7, 32)
    env = {
        "VLLM_HPU_EXPLICIT_SIGMOID_SILU": "true",
        "VLLM_HPU_EXPLICIT_SIGMOID_SILU_MIN_TOKENS": "8",
    }
    with mock.patch.dict("os.environ", env, clear=True):
        output = HPUSiluAndMul.forward_oot(x)

    gate, up = x.chunk(2, dim=-1)
    torch.testing.assert_close(output, F.silu(gate) * up)
