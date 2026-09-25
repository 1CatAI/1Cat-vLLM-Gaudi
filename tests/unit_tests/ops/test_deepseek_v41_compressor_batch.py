# SPDX-License-Identifier: Apache-2.0
"""Request-owned pair compression reuses the C1 codec and persistent state."""
import json
import os

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
if os.environ.get("DSV41_MICRO_RANK_CPUS"):
    os.sched_setaffinity(0, json.loads(os.environ["DSV41_MICRO_RANK_CPUS"])[rank])

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

torch.set_num_threads(1)
torch.hpu.set_device(rank)
torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("batch", [1, 7, 32, 64])
def test_pair_matches_c1_with_padding_ring_wrap_and_slot_reuse(batch):
    torch.manual_seed(12001 + batch)
    capacity = batch + 3
    kh = torch.randn(capacity * 8, 512).to("hpu")
    sh = (torch.randn(capacity * 8, 512) * 8).to("hpu")
    expected_kh, expected_sh = kh.clone(), sh.clone()
    positions = torch.empty(batch, dtype=torch.int32, device="hpu")
    slots = torch.empty_like(positions)
    kv = torch.empty(batch, 512, device="hpu")
    score = torch.empty_like(kv)
    pair = torch.ops.custom_op.custom_deepseek_v41_compressor_pair_bf16_gaudi2
    native = torch.ops.custom_op.custom_deepseek_v41_compressor_batch_bf16_gaudi2
    candidate = torch.compile(native, backend="hpu_backend", fullgraph=True, dynamic=False)
    owners = torch.randperm(capacity)[:batch].int()
    for generation, offset in enumerate((6, 7, 8, 9)):
        host_slots = owners.roll(generation)
        host_positions = torch.arange(batch, dtype=torch.int32) * 2 + offset
        if batch > 1:
            host_slots[-1] = -1
            host_positions[-2] = -1
        source = torch.randn(batch, 512)
        gates = torch.randn(batch, 512) * (100 if generation == 2 else 2)
        gates[:, :64] = 0
        kv.copy_(source)
        score.copy_(gates)
        positions.copy_(host_positions)
        slots.copy_(host_slots)
        reference = []
        for row, (owner, position) in enumerate(zip(host_slots.tolist(), host_positions.tolist(), strict=True)):
            if owner < 0 or position < 0:
                reference.append(torch.zeros(1, 512, dtype=torch.bfloat16))
                continue
            reference.append(
                pair(expected_kh[owner * 8:(owner + 1) * 8], expected_sh[owner * 8:(owner + 1) * 8], kv[row:row + 1],
                     score[row:row + 1], positions[row:row + 1]).cpu())
        actual = candidate(kh, sh, kv, score, positions, slots).cpu()
        assert torch.equal(actual, torch.cat(reference))
        assert torch.equal(kh.cpu().view(torch.int32), expected_kh.cpu().view(torch.int32))
        assert torch.equal(sh.cpu().view(torch.int32), expected_sh.cpu().view(torch.int32))


def test_pair_rejects_invalid_history_geometry():
    native = torch.ops.custom_op.custom_deepseek_v41_compressor_batch_bf16_gaudi2
    with pytest.raises(RuntimeError, match="slots\\*8"):
        native(torch.empty(9, 512, device="meta"), torch.empty(9, 512, device="meta"), torch.empty(2,
                                                                                                   512,
                                                                                                   device="meta"),
               torch.empty(2, 512, device="meta"), torch.empty(2, dtype=torch.int32, device="meta"),
               torch.empty(2, dtype=torch.int32, device="meta"))


@pytest.mark.parametrize("batch", [7, 64])
def test_gather_preserves_state_and_exact_fp32_pairs(batch):
    torch.manual_seed(4091)
    capacity = batch + 2
    kh, sh = torch.randn(capacity * 8, 512), torch.randn(capacity * 8, 512)
    live_kh, live_sh = kh.to("hpu"), sh.to("hpu")
    op = torch.compile(torch.ops.custom_op.custom_deepseek_v41_compressor_batch_gather_f32_gaudi2,
                       backend="hpu_backend",
                       fullgraph=True,
                       dynamic=False)
    for generation in range(3):
        kv, score = torch.randn(batch, 512), torch.randn(batch, 512)
        owners = torch.randperm(capacity)[:batch].int()
        positions = torch.arange(batch, dtype=torch.int32) * 2 + 7 + generation
        owners[-1], positions[-2] = -1, -1
        expected = torch.zeros(batch, 4, 512)
        for i, (owner, position) in enumerate(zip(owners.tolist(), positions.tolist(), strict=True)):
            if owner < 0 or position < 0:
                continue
            kh[owner * 8 + position % 8] = kv[i]
            sh[owner * 8 + position % 8] = score[i]
            start = owner * 8 + (position - position % 2) % 8
            expected[i] = torch.stack((kh[start], kh[start + 1], sh[start], sh[start + 1]))
        actual = op(live_kh, live_sh, kv.to("hpu"), score.to("hpu"), positions.to("hpu"), owners.to("hpu")).cpu()
        assert torch.equal(expected.view(torch.int32), actual.view(torch.int32))
        assert torch.equal(kh.view(torch.int32), live_kh.cpu().view(torch.int32))
        assert torch.equal(sh.view(torch.int32), live_sh.cpu().view(torch.int32))
