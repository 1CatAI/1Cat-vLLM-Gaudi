# SPDX-License-Identifier: Apache-2.0
"""Exact paired-SIMD attention against the existing device recurrence."""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
REFERENCE = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2
CANDIDATES = {
    "paired_exp": torch.ops.custom_op.custom_deepseek_v41_sparse_attn_pair_exp_bf16_gaudi2,
    "head_pair": torch.ops.custom_op.custom_deepseek_v41_sparse_attn_head_pair_bf16_gaudi2,
}
OPTIONS = {"paired_exp": (False, True), "head_pair": (False, False, True)}
SWA = torch.ops.custom_op.custom_deepseek_v41_swa_pack_write_bf16_gaudi2
FP4 = torch.ops.custom_op.custom_deepseek_v41_fp4_cache_write_bf16_gaudi2
ATTN = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2
CACHE_ATTN = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2
EVIDENCE = Path(os.environ["DSV41_RUN_EVIDENCE"])


def compare(name, expected, actual):
    dtype = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    left, right = expected.cpu().view(dtype), actual.cpu().view(dtype)
    count = int((left != right).sum())
    (EVIDENCE / f"{name}.json").write_text(json.dumps(dict(elements=left.numel(), mismatches=count)) + "\n")
    if count:
        torch.save(dict(expected=left, actual=right), EVIDENCE / f"{name}.pt")
    assert count == 0, (name, count)


@pytest.mark.parametrize("variant", ["paired_exp", "head_pair"])
@pytest.mark.parametrize("special", [False, True])
def test_recurrence_output_and_f32_statistics(special, variant):
    torch._dynamo.reset()
    torch.manual_seed(7105)
    q = torch.randn(1, 32, 512).bfloat16()
    kv = torch.randn(768, 512).bfloat16()
    sink = torch.randn(32)
    scale = torch.tensor([512**-.5])
    if special:
        bits = [0, -32768, 1, 127, 128, 16256, -16512, 32640, -128, 32704, 32705, 32767, -63, -1, 32639, -129]
        q.zero_()
        q[0, :, 0] = torch.tensor(bits * 2, dtype=torch.int16).view(torch.bfloat16)
        sink_bits = [
            0, -2147483648, 1, 8388607, 1065353216, -1082130432, 2139095040, -8388608, 2143289344, 2143289345,
            2147483647, -4194303, -1, 2139095039, -8388609, 0
        ]
        sink = torch.tensor(sink_bits * 2, dtype=torch.int32).view(torch.float32)

    def program(q, kv, ids, sink, scale, lengths):
        return REFERENCE(q, kv, ids, sink, scale, lengths), CANDIDATES[variant](q, kv, ids, sink, scale, lengths)

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    for generation, length in enumerate((-1, 0, 1, 2, 17, 127, 128, 129, 191, 256, 512, 640, 999)):
        ids = ((torch.arange(640, dtype=torch.int32) * 13 + generation) % 768).unsqueeze(0)
        ids[:, :8] = torch.tensor([-1, 0, 0, 767, 768, 9999, -1, 3], dtype=torch.int32)
        # Exercise changed cache contents and repeated/reordered slots without
        # changing the graph shape or the recurrence's index order.
        keys = kv.roll(generation, dims=0)
        expected, actual = compiled(*(x.to("hpu")
                                      for x in (q, keys, ids, sink, scale, torch.tensor([length], dtype=torch.int32))))
        for field, left, right in zip(("output", "max", "lse"), expected, actual, strict=True):
            compare(f"recurrence-{variant}-{special}-{generation}-{field}", left, right)
    torch._dynamo.reset()


# Reuse the selected-KV producer/consumer boundary cases with only the
# attention exponent implementation changed.
@pytest.mark.parametrize("variant", ["paired_exp", "head_pair"])
@pytest.mark.parametrize("main_rows,write_compressed", [(0, False), (256, False), (256, True), (512, True)])
def test_ordered_cache_consumer(main_rows, write_compressed, variant):
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
            actual = CACHE_ATTN(q, swa, main, ids, sink, scale, lengths, swa_done, fp4_done, True, *OPTIONS[variant])
        else:
            expected = ATTN(q, swa, main, ids, sink, scale, lengths, swa_done, True, False)
            actual = ATTN(q, swa, main, ids, sink, scale, lengths, swa_done, True, *OPTIONS[variant])
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
        compare(f"consumer-{variant}-{main_rows}-{write_compressed}-{generation}", expected, actual)
    torch._dynamo.reset()


def test_head_pair_batch_mapping():
    torch._dynamo.reset()
    torch.manual_seed(7106)
    inputs = (torch.randn(2, 2,
                          512).bfloat16(), torch.randn(64,
                                                       512).bfloat16(), torch.arange(32,
                                                                                     dtype=torch.int32).repeat(2, 1),
              torch.randn(2), torch.tensor([512**-.5]), torch.tensor([13, 31], dtype=torch.int32))
    inputs[2][1, :4] = torch.tensor([-1, 0, 0, 99], dtype=torch.int32)

    def program(*args):
        return REFERENCE(*args), CANDIDATES["head_pair"](*args)

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    expected, actual = compiled(*(x.to("hpu") for x in inputs))
    for field, left, right in zip(("output", "max", "lse"), expected, actual, strict=True):
        compare("head-pair-batch-" + field, left, right)
    torch._dynamo.reset()
