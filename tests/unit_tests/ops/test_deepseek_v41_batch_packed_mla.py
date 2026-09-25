# SPDX-License-Identifier: Apache-2.0
"""Fused packed decoding retains MLA arithmetic and request write dependencies."""
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

from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_packed_mla, write_state_rows  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa  # noqa: E402

torch.set_num_threads(1)
torch.hpu.set_device(rank)
torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


def bits(tensor):
    return tensor.cpu().view(torch.int16)


@pytest.mark.parametrize("batch,width", [(1, 64), (4, 128), (8, 640)])
def test_native_packed_mla_local_and_physical_rows(batch, width):
    torch._dynamo.reset()
    torch.manual_seed(3491)
    q = torch.randn(batch, 32, 512).bfloat16().to("hpu")
    swa = pack_swa(torch.randn(512, 512).bfloat16()).to("hpu")
    main = pack_fp4(torch.randn(1024, 512).bfloat16(), 16).to("hpu")
    # Deliberately break identity mapping, retain duplicates, and distinguish
    # invalid physical rows (zero KV, valid mask) from invalid local indices.
    rows_cpu = torch.randint(-9, 1550, (1, 2 * width), dtype=torch.int32)
    ids_cpu = torch.randint(-5, 2 * width + 4, (batch, width), dtype=torch.int32)
    ids_cpu[:, 3::11] = 2
    rows_cpu[0, 2] = -1
    rows, ids = rows_cpu.to("hpu"), ids_cpu.to("hpu")
    lengths = torch.full((batch, ), width, dtype=torch.int32, device="hpu")
    sink = torch.randn(32, device="hpu")
    scale = torch.tensor([512**-0.5], device="hpu")
    old = torch.ops.custom_op.custom_deepseek_v41_batch_paged_mla_mme_gaudi2
    new = (torch.ops.custom_op.custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2
           if os.environ.get("DSV41_TEST_PACKED_MLA_VECTOR") == "1" else
           torch.ops.custom_op.custom_deepseek_v41_batch_packed_sram_mla_mme_gaudi2
           if os.environ.get("DSV41_TEST_PACKED_MLA_SRAM") == "1" else
           torch.ops.custom_op.custom_deepseek_v41_batch_packed_mla_mme_gaudi2)
    old = torch.compile(old, backend="hpu_backend", fullgraph=True, dynamic=False)
    new = torch.compile(new, backend="hpu_backend", fullgraph=True, dynamic=False)
    for change in range(3):
        q.copy_(torch.randn(batch, 32, 512).bfloat16())
        rows.copy_(rows_cpu.roll(change * 7, 1))
        ids.copy_(ids_cpu.roll(change, 0))
        lengths.copy_(
            torch.tensor([width, 0, 3, width - 1] *
                         ((batch + 3) //
                          4), dtype=torch.int32)[:batch] if change else torch.full((batch, ), width, dtype=torch.int32))
        args = (q, swa, main, rows, ids, sink, scale, lengths)
        expected, actual = bits(old(*args)), bits(new(*args))
        assert torch.equal(actual, expected), (batch, width, change, int((actual != expected).sum()))


@pytest.mark.parametrize("batch,ratio", [(1, 0), (4, 1), (8, 2), (16, 0), (32, 2), (64, 1), (5, 2)])
def test_request_write_gather_mla_pages_slots_and_padding(batch, ratio):
    torch._dynamo.reset()
    torch.manual_seed(3517)
    capacity = max(8, batch)
    swa_cpu = pack_swa(torch.randn(capacity * 256, 512).bfloat16())
    swa = swa_cpu.to("hpu")
    main = pack_fp4(torch.randn((capacity * 8 + 1) * 128, 512).bfloat16(), 16).to("hpu")
    slots_cpu = torch.randperm(capacity)[:batch].int()
    if batch > 1:
        slots_cpu[-1] = -1
    positions_cpu = torch.tensor([0, 127, 255, 256, 511, 512, 999, 1023] * ((batch + 7) // 8),
                                 dtype=torch.int32)[:batch]
    pages_cpu = torch.stack([torch.arange(8) + 1 + int(max(0, slot)) * 8 for slot in slots_cpu]).int()
    selected_cpu = torch.arange(512).int().expand(batch, -1).clone()
    selected_cpu[:, 13::17] = -1
    selected_cpu[:, 15::19] = 0
    positions, slots, pages, selected = (x.to("hpu") for x in (positions_cpu, slots_cpu, pages_cpu, selected_cpu))
    q = torch.randn(batch, 32, 512).bfloat16().to("hpu")
    values = torch.randn(batch, 512).bfloat16().to("hpu")
    sink = torch.randn(32, device="hpu")
    scale = torch.tensor([512**-0.5], device="hpu")
    done = torch.full((batch, ), -1, dtype=torch.int32, device="hpu")

    def make(fused):

        def chain(query, cache, kv, ids, tables, pos, slot):
            write_rows = torch.where((slot >= 0) & (pos >= 0), slot * 256 + pos.remainder(256), -1).int()
            write_done = write_state_rows(cache, pack_swa(kv), write_rows)
            return batch_packed_mla(query,
                                    cache,
                                    main,
                                    ids,
                                    tables,
                                    pos,
                                    slot,
                                    sink,
                                    scale,
                                    write_done,
                                    done,
                                    ratio=ratio,
                                    fused_gather=fused,
                                    sram_gather=fused and os.environ.get("DSV41_TEST_PACKED_MLA_SRAM") == "1",
                                    vector_gather=fused and os.environ.get("DSV41_TEST_PACKED_MLA_VECTOR") == "1")

        return torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)

    old, new = make(False), make(True)
    for change in range(3):
        q.copy_(torch.randn(batch, 32, 512).bfloat16())
        values.copy_(torch.randn(batch, 512).bfloat16())
        positions.copy_((positions_cpu + change * 129).remainder(1024))
        slots.copy_(slots_cpu.roll(change))
        pages.copy_(pages_cpu.roll(change, 0))
        selected.copy_(selected_cpu.roll(change * 3, 1))
        args = (q, swa, values, selected, pages, positions, slots)
        swa.copy_(swa_cpu)
        expected, expected_state = bits(old(*args)), swa.cpu()
        swa.copy_(swa_cpu)
        actual, actual_state = bits(new(*args)), swa.cpu()
        assert torch.equal(actual, expected), (batch, ratio, change, int((actual != expected).sum()))
        assert torch.equal(actual_state, expected_state), (batch, ratio, change, "state")
        if batch > 1:
            assert not actual[slots_cpu.roll(change) < 0].any(), "Padding query produced output"
