# SPDX-License-Identifier: Apache-2.0
"""Opt-in checks of group-major state mutation on Gaudi."""
import os

import pytest
import torch

from flashinfer_gaudi._reference import qwen38_fused_decode_step_direct
from vllm_gaudi.v1.worker.hpu_model_runner import _zero_compact_gdn_slot

pytestmark = pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_RUN_HARDWARE_TESTS") != "1",
                                reason="explicit Gaudi2 hardware test opt-in required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_hpu_reused_slot_clear_preserves_other_group_major_rows(dtype):
    original = torch.arange(18 * 6).reshape(18, 2, 3).to(dtype) + 1
    for slot in (0, 3, 7):
        pool = original.to("hpu")
        _zero_compact_gdn_slot([pool], slot, 2, 8)
        expected = original.clone()
        expected[[0, slot + 1, 8 + slot + 1, 17]] = 0
        torch.testing.assert_close(pool.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("active,padded", [(3, 4), (7, 8)])
def test_compiled_padded_decode_is_row_local_and_reentrant(active, padded):
    generator = torch.Generator().manual_seed(203)
    capacity, groups, group_offset = 8, 2, 1
    start = group_offset * capacity + 1

    def rand(*shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=generator, dtype=dtype) * 0.01).to("hpu")

    def run(packed, a, b, a_log, dt_bias, weight, conv_pool, state_pool):
        batch = packed.shape[0]
        return qwen38_fused_decode_step_direct(
            packed,
            a,
            b,
            a_log,
            dt_bias,
            conv_pool.narrow(0, start, batch),
            weight,
            None,
            state_pool.narrow(0, start, batch),
            128**-0.5,
        )[0]

    compiled = torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)
    conv = rand(capacity * groups + 2, 3, 10240)
    state = rand(capacity * groups + 2, 48, 128, 128, dtype=torch.float32)
    reference_conv, reference_state = conv.clone(), state.clone()
    a_log = rand(48, dtype=torch.float32) - 2
    dt_bias, weight = rand(48), rand(10240, 4)
    before_conv, before_state = conv.cpu(), state.cpu()
    # Re-enter the same static graph with changed inputs. Padding is deliberately
    # non-finite to prove that it cannot contaminate any active request row.
    for _ in range(3):
        packed, a, b = rand(padded, 10240), rand(padded, 48), rand(padded, 48)
        reference_packed = packed.clone()
        reference_packed[active:] = 0
        packed[active:] = float("nan")
        actual = compiled(packed, a, b, a_log, dt_bias, weight, conv, state)
        # Use the same compiled shape to isolate padding, not unrelated
        # eager/compiled or different-batch BF16 rounding differences.
        expected = compiled(reference_packed, a, b, a_log, dt_bias, weight, reference_conv, reference_state)
        torch.hpu.synchronize()
        torch.testing.assert_close(actual[:active].cpu(), expected[:active].cpu(), atol=0, rtol=0)
        torch.testing.assert_close(conv[start:start + active].cpu(),
                                   reference_conv[start:start + active].cpu(),
                                   atol=0,
                                   rtol=0)
        torch.testing.assert_close(state[start:start + active].cpu(),
                                   reference_state[start:start + active].cpu(),
                                   atol=0,
                                   rtol=0)
    untouched = [*range(start), *range(start + padded, state.shape[0])]
    torch.testing.assert_close(conv.cpu()[untouched], before_conv[untouched], atol=0, rtol=0)
    torch.testing.assert_close(state.cpu()[untouched], before_state[untouched], atol=0, rtol=0)
    # Reclaim a dirty padding slot without clearing any of the active rows.
    active_before = state[start:start + active].cpu()
    _zero_compact_gdn_slot([conv, state], active, groups, capacity)
    assert torch.isfinite(state[start + active].cpu()).all()
    torch.testing.assert_close(state[start:start + active].cpu(), active_before, atol=0, rtol=0)
