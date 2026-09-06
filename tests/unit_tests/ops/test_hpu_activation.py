# SPDX-License-Identifier: Apache-2.0

import torch

import vllm_gaudi.ops.hpu_activation as hpu_activation
from vllm_gaudi.ops.hpu_activation import HPUSiluAndMul


def test_hpu_silu_and_mul_uses_triton_result(default_vllm_config, monkeypatch):
    input_tensor = torch.randn(2, 512, dtype=torch.bfloat16)
    expected = torch.full((2, 256), 3.0, dtype=torch.bfloat16)
    monkeypatch.setattr(hpu_activation, "_triton_silu_and_mul", lambda _: expected)

    actual = HPUSiluAndMul().forward_oot(input_tensor)

    assert actual is expected


def test_hpu_silu_and_mul_preserves_vendor_fallback(default_vllm_config, monkeypatch):
    input_tensor = torch.randn(2, 512, dtype=torch.bfloat16)
    monkeypatch.setattr(hpu_activation, "_triton_silu_and_mul", lambda _: None)

    actual = HPUSiluAndMul().forward_oot(input_tensor)
    expected = HPUSiluAndMul.forward_native(input_tensor)

    torch.testing.assert_close(actual, expected)
