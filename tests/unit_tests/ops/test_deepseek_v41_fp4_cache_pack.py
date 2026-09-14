# SPDX-License-Identifier: Apache-2.0
"""Exact FP4 bytes, scratch-row ownership and ordered C1 cache consumers."""

import json
import os
from pathlib import Path

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Native FP4 cache tests require an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
WRITE = torch.ops.custom_op.custom_deepseek_v41_fp4_cache_write_bf16_gaudi2
SWA_WRITE = torch.ops.custom_op.custom_deepseek_v41_swa_pack_write_bf16_gaudi2
ATTENTION = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2
OLD_ATTENTION = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_lengths_gaudi2
EVIDENCE = Path(os.environ["DSV41_RUN_EVIDENCE"])


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    # OpOverloadPacket wrappers share a code object across independent shape cohorts.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("group", (16, 32))
def test_fp4_pack_exact_checkpoint_bytes(group):
    op = getattr(torch.ops.custom_op, f"custom_deepseek_v41_fp4_pack_g{group}_bf16_gaudi2")
    compiled = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)

    def original(value):
        return pack_fp4(value, group)

    reference = torch.compile(original, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(4190 + group)
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    cases = [("all_bf16_group_maxima", encodings.repeat_interleave(group).reshape(-1, 512))]
    mixed = torch.randn(256, 512).bfloat16()
    mixed[:, ::group] = torch.exp2(torch.arange(-126, 130, dtype=torch.float32)).clamp_max(2.**127).unsqueeze(1)
    mixed[:, 1::group], mixed[:, 2::group] = 0., -0.
    cases.append(("mixed_scale_boundaries", mixed))
    special = torch.randn(32, 512).bfloat16()
    special[0::4, 0], special[1::4, 0], special[2::4, 0] = float("nan"), float("inf"), -float("inf")
    special[3::4, :group] = -0.
    cases.append(("mixed_nonfinite", special))
    ties = torch.tensor([6., .25, -.25, .75, -.75, 1.25, -1.25, 1.75, -1.75,
                         2.5, -2.5, 3.5, -3.5, 5., -5., -0.]).repeat(32).reshape(1, 512).bfloat16()
    cases.extend((("rounding_ties", ties), ("actual_c1", torch.randn(1, 512 if group == 16 else 128).bfloat16())))
    results = []
    for name, value in cases:
        device = value.to("hpu")
        expected, actual = reference(device).cpu(), compiled(device).cpu()
        eager = op(device).cpu()
        mismatch = (actual != expected).nonzero()
        item = {"case": name, "shape": list(value.shape), "byte_mismatches": len(mismatch),
                "ordinary_compiled_exact": torch.equal(actual, eager)}
        if len(mismatch):
            torch.save({"input": value, "actual": actual, "expected": expected},
                       EVIDENCE / f"fp4-g{group}-{name}.pt")
            item["first_mismatch"] = mismatch[:12].tolist()
        results.append(item)
    (EVIDENCE / f"fp4-g{group}-results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert all(r["byte_mismatches"] == 0 and r["ordinary_compiled_exact"] for r in results), results


def test_fp4_write_orders_consumers_and_preserves_scratch_rows():
    torch.manual_seed(4191)
    sink = torch.randn(32, dtype=torch.float32, device="hpu")
    scale = torch.tensor([512**-0.5], device="hpu")

    def program(q, swa, main, index, kv, main_value, index_value, position, slot, ids, lengths):
        swa_done = SWA_WRITE(swa, kv, position)
        fp4_done = WRITE(main, index, main_value, index_value, slot)
        output = ATTENTION(q, swa, main[:256], ids, sink, scale, lengths, swa_done, fp4_done)
        return output, fp4_done

    def old(q, swa, main, index, kv, main_value, index_value, position, slot, ids, lengths):
        swa.index_copy_(0, position.long(), pack_swa(kv))
        main.index_copy_(0, slot.long(), pack_fp4(main_value, 16))
        index.index_copy_(0, slot.long(), pack_fp4(index_value, 32))
        return OLD_ATTENTION(q, swa, main[:256], ids, sink, scale, lengths)

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = torch.compile(old, backend="hpu_backend", fullgraph=True, dynamic=False)
    initial = (pack_swa(torch.randn(512, 512).bfloat16()), pack_fp4(torch.randn(768, 512).bfloat16(), 16),
               pack_fp4(torch.randn(768, 128).bfloat16(), 32))
    caches = [v.to("hpu") for v in initial]
    reference_caches = [v.to("hpu") for v in initial]
    pointers = [v.data_ptr() for v in caches]
    results = []
    positions = (0, 1, 127, 128, 255, 511, 0, 127, 128)
    slots = (0, 1, 255, 256, 511, 767, 0, 127, 128)
    for generation, (pos, slot) in enumerate(zip(positions, slots, strict=True)):
        offsets = torch.arange(pos - 127, pos + 1, dtype=torch.int32)
        swa_ids = torch.where(offsets >= 0, offsets, -1)
        extra = [512 + slot, 512 + ((slot - 1) % 256), -1, 512 + slot] if slot < 256 else [512, -1, -1, -1]
        ids = torch.cat((swa_ids, torch.tensor(extra, dtype=torch.int32))).unsqueeze(0).to("hpu")
        inputs = [torch.randn(1, width).bfloat16().to("hpu") for width in (512, 512, 128)]
        position = torch.tensor([pos], dtype=torch.int32, device="hpu")
        slot_tensor = torch.tensor([slot], dtype=torch.int32, device="hpu")
        lengths = torch.tensor([132], dtype=torch.int32, device="hpu")
        q = torch.randn(1, 32, 512).bfloat16().to("hpu")
        expected = reference(q, *reference_caches, *inputs, position, slot_tensor, ids, lengths).cpu()
        output, completion = compiled(q, *caches, *inputs, position, slot_tensor, ids, lengths)
        actual = output.cpu()
        states_equal = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(caches, reference_caches, strict=True))
        equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        complete = torch.equal(completion.cpu(), torch.full((36,), slot, dtype=torch.int32))
        results.append({"generation": generation, "position": pos, "slot": slot, "states_equal": states_equal,
                        "attention_bitwise_equal": equal, "complete": complete,
                        "addresses_stable": pointers == [v.data_ptr() for v in caches]})
        if not states_equal or not equal:
            torch.save({"actual": actual, "expected": expected, "main": caches[1].cpu(),
                        "reference_main": reference_caches[1].cpu(), "index": caches[2].cpu(),
                        "reference_index": reference_caches[2].cpu()}, EVIDENCE / f"fp4-write-{generation}.pt")
    (EVIDENCE / "fp4-write-results.json").write_text(json.dumps(results, indent=2) + "\n")
    assert all(r["states_equal"] and r["attention_bitwise_equal"] and r["complete"] and
               r["addresses_stable"] for r in results), results


def test_fp4_write_invalid_slot_does_not_touch_storage():
    main = torch.full((8, 288), 173, dtype=torch.uint8, device="hpu")
    index = torch.full((8, 68), 174, dtype=torch.uint8, device="hpu")
    values = [torch.zeros(1, width, dtype=torch.bfloat16, device="hpu") for width in (512, 128)]
    for slot in (-1, 8):
        position = torch.tensor([slot], dtype=torch.int32, device="hpu")
        completion = WRITE(main, index, *values, position).cpu()
        assert torch.equal(completion, torch.full((36,), -1, dtype=torch.int32))
        assert torch.equal(main.cpu(), torch.full((8, 288), 173, dtype=torch.uint8))
        assert torch.equal(index.cpu(), torch.full((8, 68), 174, dtype=torch.uint8))
