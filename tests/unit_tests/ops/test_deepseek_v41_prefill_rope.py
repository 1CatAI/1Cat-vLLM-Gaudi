# SPDX-License-Identifier: Apache-2.0
"""Preserve prefill products and the distinct small-query RoPE contract."""
import os
from types import SimpleNamespace

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Native RoPE checks require an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import _apply_rope_torch, rotary_table  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("tokens", [7, 8, 127, 8191, 8192])
def test_changed_positions_and_normal_prefill_dispatch(monkeypatch, tokens):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_ROPE", "1")
    table_cpu = rotary_table(64, 32768, 10000)
    table = table_cpu.to("hpu")
    native = torch.cat((table_cpu[..., 0], table_cpu[..., 1]), -1).contiguous().to("hpu")
    owner = SimpleNamespace(native_rope=True,
                            rotary=table,
                            rotary_native=native,
                            _rotary_table=lambda: table,
                            _rotary_native_table=lambda: native)
    values = torch.empty(tokens, 1, 128, device="hpu", dtype=torch.bfloat16)
    positions = torch.empty(tokens, device="hpu", dtype=torch.int32)
    for generation in range(2):
        values.copy_(torch.randn(tokens, 1, 128).bfloat16())
        # Includes reordered requests/positions; no history-dependent signature.
        positions.copy_((torch.arange(tokens, dtype=torch.int32).flip(0) + generation * 8192) % 32768)
        for inverse in (False, True):
            expected = _apply_rope_torch(values, positions, table, inverse).cpu().view(torch.int16)
            for cls in (CSA2Attention, PagedCSA2Attention):
                actual = cls._rope(owner, values, positions, inverse).cpu().view(torch.int16)
                assert torch.equal(actual, expected)


def test_small_decode_retains_its_previous_dispatch(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_ROPE", "1")
    owner = SimpleNamespace(native_rope=True, _rotary_native_table=lambda: torch.empty(8, 64, device="hpu"))
    for tokens in (1, 6):
        value = torch.randn(tokens, 1, 128).bfloat16().to("hpu")
        positions = torch.arange(tokens, device="hpu", dtype=torch.int32)
        for inverse in (False, True):
            called = []
            name = "custom_deepseek_v41_rope_" + ("inverse_" if inverse else "") + "bf16_gaudi2"

            def old(x, p, table, called=called):
                called.append(x.shape)
                return x

            with monkeypatch.context() as patch:
                patch.setattr(torch.ops.custom_op, name, old)
                assert PagedCSA2Attention._rope(owner, value, positions, inverse) is not None
                assert called == [value.shape]
