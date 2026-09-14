# SPDX-License-Identifier: Apache-2.0
"""Raw checkpoint bytes and ordered cache consumption on a leased Gaudi2."""

import json
import os
from pathlib import Path

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Native SWA tests require an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_math import pack_swa  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
PACK = torch.ops.custom_op.custom_deepseek_v41_swa_pack_bf16_gaudi2
WRITE = torch.ops.custom_op.custom_deepseek_v41_swa_pack_write_bf16_gaudi2
ATTENTION = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2
OLD_ATTENTION = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_lengths_gaudi2
EVIDENCE = Path(os.environ["DSV41_RUN_EVIDENCE"])


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    # Independent kernel cohorts must not share Dynamo's wrapper specialization count.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_swa_pack_exact_checkpoint_bytes():
    compiled = torch.compile(PACK, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = torch.compile(pack_swa, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(4118)
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    cases = [("all_bf16_group_maxima", encodings.repeat_interleave(32).reshape(4096, 512))]
    mixed = torch.randn(256, 512).bfloat16()
    mixed[:, ::32] = torch.exp2(torch.arange(-126, 130, dtype=torch.float32)).clamp_max(2.**127).unsqueeze(1)
    mixed[:, 1::32], mixed[:, 2::32] = 0., -0.
    cases.append(("mixed_scale_boundaries", mixed))
    special = torch.randn(32, 512).bfloat16()
    special[0::4, 0], special[1::4, 0], special[2::4, 0] = float("nan"), float("inf"), -float("inf")
    special[3::4, :32] = -0.
    cases.extend((("mixed_nonfinite", special), ("actual_c1", torch.randn(1, 512).bfloat16())))
    results = []
    for name, value in cases:
        device = value.to("hpu")
        expected = reference(device).cpu()
        actual = compiled(device).cpu()
        eager = PACK(device).cpu()
        mismatches = (actual != expected).nonzero()
        item = {"case": name, "shape": list(value.shape), "byte_mismatches": len(mismatches),
                "ordinary_compiled_exact": torch.equal(actual, eager)}
        if len(mismatches):
            torch.save({"input": value, "actual": actual, "expected": expected}, EVIDENCE / f"{name}.pt")
            item["first_mismatch"] = mismatches[:12].tolist()
        results.append(item)
    (EVIDENCE / "swa-byte-results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert all(r["byte_mismatches"] == 0 and r["ordinary_compiled_exact"] for r in results), results


def test_swa_write_preserves_other_rows_and_orders_attention():
    torch.manual_seed(4119)
    sink = torch.randn(32, dtype=torch.float32, device="hpu")
    scale = torch.tensor([512**-0.5], device="hpu")

    def program(q, cache, value, position, indices, lengths):
        completion = WRITE(cache, value, position)
        output = ATTENTION(q, cache, cache, indices, sink, scale, lengths, completion)
        return output, completion

    def old(q, cache, value, position, indices, lengths):
        cache.index_copy_(0, position.long(), pack_swa(value))
        return OLD_ATTENTION(q, cache, cache, indices, sink, scale, lengths)

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = torch.compile(old, backend="hpu_backend", fullgraph=True, dynamic=False)
    initial = pack_swa(torch.randn(512, 512).bfloat16())
    cache, expected_cache = initial.to("hpu"), initial.to("hpu")
    pointer = cache.data_ptr()
    results = []
    for generation, slot in enumerate((0, 1, 31, 127, 128, 255, 511, 0, 1, 126, 127, 128)):
        position = torch.tensor([slot], dtype=torch.int32, device="hpu")
        offsets = torch.arange(slot - 127, slot + 1, dtype=torch.int32)
        indices = torch.where(offsets >= 0, offsets, -1).unsqueeze(0).to("hpu")
        lengths = torch.tensor([128], dtype=torch.int32, device="hpu")
        q = torch.randn(1, 32, 512).bfloat16().to("hpu")
        value = torch.randn(1, 512).bfloat16().to("hpu")
        expected = reference(q, expected_cache, value, position, indices, lengths).cpu()
        output, completion = compiled(q, cache, value, position, indices, lengths)
        actual = output.cpu()
        state_equal = torch.equal(cache.cpu(), expected_cache.cpu())
        output_equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        complete = torch.equal(completion.cpu(), torch.full((16,), slot, dtype=torch.int32))
        results.append({"generation": generation, "position": slot, "cache_equal": state_equal,
                        "attention_bitwise_equal": output_equal, "complete": complete,
                        "cache_address_stable": cache.data_ptr() == pointer})
        if not state_equal or not output_equal:
            torch.save({"actual": actual, "expected": expected, "cache": cache.cpu(),
                        "expected_cache": expected_cache.cpu()}, EVIDENCE / f"write-{generation}.pt")
    (EVIDENCE / "swa-write-results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert all(r["cache_equal"] and r["attention_bitwise_equal"] and r["complete"] and
               r["cache_address_stable"] for r in results), results


def test_swa_write_invalid_slot_does_not_touch_storage():
    cache = torch.full((8, 528), 173, dtype=torch.uint8, device="hpu")
    value = torch.zeros(1, 512, dtype=torch.bfloat16, device="hpu")
    for slot in (-1, 8):
        position = torch.tensor([slot], dtype=torch.int32, device="hpu")
        completion = WRITE(cache, value, position).cpu()
        assert torch.equal(completion, torch.full((16,), -1, dtype=torch.int32))
        assert torch.equal(cache.cpu(), torch.full((8, 528), 173, dtype=torch.uint8))


def test_swa_pack_rejects_incompatible_contracts():
    with pytest.raises(RuntimeError, match="BF16"):
        PACK(torch.empty((1, 512), dtype=torch.float32, device="meta"))
    with pytest.raises(RuntimeError, match="divisible"):
        PACK(torch.empty((1, 511), dtype=torch.bfloat16, device="meta"))
    with pytest.raises(RuntimeError, match="C1"):
        WRITE(torch.empty((8, 528), dtype=torch.uint8, device="meta"),
              torch.empty((2, 512), dtype=torch.bfloat16, device="meta"),
              torch.empty((2,), dtype=torch.int32, device="meta"))
