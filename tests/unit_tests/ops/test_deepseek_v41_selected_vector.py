# SPDX-License-Identifier: Apache-2.0
"""Exact selected-row encoding and ordered consumer checks on a leased HPU."""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Selected-vector checks require an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
GATHER = torch.ops.custom_op.custom_deepseek_v41_selected_kv_bf16_gaudi2
SWA = torch.ops.custom_op.custom_deepseek_v41_swa_pack_write_bf16_gaudi2
FP4 = torch.ops.custom_op.custom_deepseek_v41_fp4_cache_write_bf16_gaudi2
ATTN = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2
CACHE_ATTN = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2
EVIDENCE = Path(os.environ["DSV41_RUN_EVIDENCE"])


def compare(name, expected, actual):
    expected, actual = expected.cpu(), actual.cpu()
    left = expected.view(torch.int16) if expected.dtype == torch.bfloat16 else expected
    right = actual.view(torch.int16) if actual.dtype == torch.bfloat16 else actual
    mismatch = int((left != right).sum())
    record = {"name": name, "elements": left.numel(), "mismatches": mismatch}
    (EVIDENCE / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    if mismatch:
        torch.save({"expected": expected, "actual": actual}, EVIDENCE / f"{name}.pt")
    assert mismatch == 0, record


@pytest.mark.parametrize("kind", ["swa", "main"])
def test_all_encoding_pairs(kind):
    torch._dynamo.reset()
    swa = torch.zeros(512, 528, dtype=torch.uint8)
    main = torch.zeros(256, 288, dtype=torch.uint8)
    if kind == "swa":
        swa[:256, :512] = torch.arange(256, dtype=torch.uint8).repeat(2)
        swa[:256, 512:] = torch.arange(256, dtype=torch.uint8).unsqueeze(1)
        ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
    else:
        nibble = torch.arange(512, dtype=torch.int32) % 16
        main[:, :256] = (nibble[::2] | (nibble[1::2] << 4)).to(torch.uint8)
        main[:, 256:] = torch.arange(256, dtype=torch.uint8).unsqueeze(1)
        ids = (512 + torch.arange(256, dtype=torch.int32)).unsqueeze(0)

    def program(swa, main, ids):
        return GATHER(swa, main, ids, False), GATHER(swa, main, ids, True)

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    expected, actual = compiled(swa.to("hpu"), main.to("hpu"), ids.to("hpu"))
    compare(f"encoding-{kind}-rows", expected[0], actual[0])
    compare(f"encoding-{kind}-indices", expected[1], actual[1])
    torch._dynamo.reset()


@pytest.mark.parametrize("slots", [1, 127, 128, 129, 640, 1024])
def test_slot_tail_alias_and_changes(slots):
    torch._dynamo.reset()
    torch.manual_seed(4200 + slots)
    swa = pack_swa(torch.randn(512, 512).bfloat16()).to("hpu")
    main = pack_fp4(torch.randn(256, 512).bfloat16(), 16).to("hpu")

    def program(swa, main, ids):
        return GATHER(swa, main, ids, False), GATHER(swa, main, ids, True)

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    for generation in range(3):
        ids = ((torch.arange(slots, dtype=torch.int32) * (generation + 3)) % 800 - 1).unsqueeze(0)
        ids[:, ::7] = 511
        ids[:, ::11] = 512
        ids[:, ::13] = -1
        active_main = swa if generation == 2 else main
        expected, actual = compiled(swa, active_main, ids.to("hpu"))
        compare(f"slots-{slots}-{generation}-rows", expected[0], actual[0])
        compare(f"slots-{slots}-{generation}-indices", expected[1], actual[1])
    torch._dynamo.reset()


@pytest.mark.parametrize("main_rows,write_compressed", [(0, False), (256, False), (256, True), (512, True)])
def test_ordered_cache_consumer(main_rows, write_compressed):
    torch._dynamo.reset()
    torch.manual_seed(4300 + main_rows)
    swa = pack_swa(torch.randn(512, 512).bfloat16()).to("hpu")
    main = pack_fp4(torch.randn(main_rows, 512).bfloat16(), 16).to("hpu") if main_rows else swa
    index = pack_fp4(torch.randn(max(main_rows, 1), 128).bfloat16(), 32).to("hpu")
    sink = torch.randn(32, dtype=torch.float32, device="hpu")
    scale = torch.tensor([512**-.5], device="hpu")

    def program(q, kv, main_value, index_value, position, slot, ids, lengths):
        swa_done = SWA(swa, kv, position)
        if write_compressed:
            fp4_done = FP4(main, index, main_value, index_value, slot)
            expected = CACHE_ATTN(q, swa, main, ids, sink, scale, lengths, swa_done, fp4_done, True, False)
            actual = CACHE_ATTN(q, swa, main, ids, sink, scale, lengths, swa_done, fp4_done, True, True)
        else:
            expected = ATTN(q, swa, main, ids, sink, scale, lengths, swa_done, True, False)
            actual = ATTN(q, swa, main, ids, sink, scale, lengths, swa_done, True, True)
        return expected, actual

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    for generation, pos in enumerate((0, 1, 127, 128, 255, 256, 510, 511, 0)):
        window = torch.arange(pos - 127, pos + 1, dtype=torch.int32)
        ids = torch.where(window >= 0, window, -1)
        visible = min(pos + 1, main_rows)
        if main_rows:
            rows = torch.arange(512, dtype=torch.int32)
            ids = torch.cat((ids, torch.where(rows < visible, rows + 512, -1)))
        if generation == 0:
            ids.fill_(-1)
        elif generation % 3 == 0:
            ids[:8] = torch.tensor([0, 0, -1, 0, 511, -1, 511, 0], dtype=torch.int32)
        position = torch.tensor([pos], dtype=torch.int32, device="hpu")
        slot = torch.tensor([min(pos, max(main_rows, 1) - 1)], dtype=torch.int32, device="hpu")
        lengths = torch.tensor([128 + visible if main_rows else 128], dtype=torch.int32, device="hpu")
        values = [torch.randn(1, width).bfloat16().to("hpu") for width in (512, 512, 128)]
        query = torch.randn(1, 32, 512).bfloat16().to("hpu")
        expected, actual = compiled(query, *values, position, slot, ids.unsqueeze(0).to("hpu"), lengths)
        compare(f"consumer-{main_rows}-{write_compressed}-{generation}", expected, actual)
    torch._dynamo.reset()
