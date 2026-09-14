# SPDX-License-Identifier: Apache-2.0
"""Incremental KV writes against the frozen packed device consumer."""
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
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4, unpack_swa  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
OPS = torch.ops.custom_op
EVIDENCE = Path(os.environ["DSV41_RUN_EVIDENCE"])


def exact(name, left, right):
    left, right = left.cpu(), right.cpu()
    if left.dtype == torch.bfloat16:
        left, right = left.view(torch.int16), right.view(torch.int16)
    elif left.dtype == torch.float32:
        left, right = left.view(torch.int32), right.view(torch.int32)
    count = int((left != right).sum())
    (EVIDENCE / f"{name}.json").write_text(json.dumps(dict(elements=left.numel(), mismatches=count)) + "\n")
    if count:
        torch.save(dict(expected=left, actual=right), EVIDENCE / f"{name}.pt")
    assert count == 0, (name, count)


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("offset,main_rows", [(0, 0), (512, 256), (0, 512)])
def test_incremental_bytes_and_ordered_attention(offset, main_rows):
    torch.manual_seed(7710 + main_rows)
    initial_swa = pack_swa(torch.randn(512, 512).bfloat16())
    initial_main = pack_fp4(torch.randn(768, 512).bfloat16(), 16)
    initial_index = pack_fp4(torch.randn(768, 128).bfloat16(), 32)
    caches = [x.to("hpu") for x in (initial_swa, initial_main, initial_index)]
    refs = [x.to("hpu") for x in (initial_swa, initial_main, initial_index)]
    shadow = torch.full((1024, 512), 17., dtype=torch.bfloat16, device="hpu")
    # Use the same prefill compatibility decoder that initializes model state.
    shadow[offset:offset + 512].copy_(unpack_swa(caches[0]))
    main_shadow = unpack_fp4(caches[1]).contiguous()
    sink = torch.randn(32, device="hpu")
    scale = torch.tensor([512**-.5], device="hpu")

    def program(q, kv, mv, iv, pos, slot, ids, lengths):
        old_swa = OPS.custom_deepseek_v41_swa_pack_write_bf16_gaudi2(refs[0], kv, pos)
        new_swa = OPS.custom_deepseek_v41_swa_decoded_write_bf16_gaudi2(caches[0], kv, pos, shadow, offset)
        if main_rows:
            old_main = OPS.custom_deepseek_v41_fp4_cache_write_bf16_gaudi2(refs[1], refs[2], mv, iv, slot)
            new_main = OPS.custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2(caches[1], caches[2], mv, iv, slot,
                                                                             main_shadow)
            expected = OPS.custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2(
                q, refs[0], refs[1][:main_rows], ids, sink, scale, lengths, old_swa, old_main, True)
        else:
            new_main = new_swa
            expected = OPS.custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2(
                q, refs[0], refs[0], ids, sink, scale, lengths, old_swa, True)
        actual = OPS.custom_deepseek_v41_decoded_attn_bf16_gaudi2(q, shadow, main_shadow, ids, sink, scale, lengths,
                                                                  new_swa, new_main, offset, main_rows)[0]
        return expected, actual, new_swa, new_main

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    for generation, (pos,
                     slot) in enumerate(zip((0, 1, 127, 128, 255, 511, 0), (0, 1, 255, 256, 511, 767, 0), strict=True)):
        ids = (torch.arange(640, dtype=torch.int32) * 13 + generation) % (512 + main_rows)
        ids[:8] = torch.tensor([-1, pos, pos, 511, 512, 512 + main_rows, 9999, -1], dtype=torch.int32)
        inputs = [torch.randn(1, width).bfloat16().to("hpu") for width in (512, 512, 128)]
        query = torch.randn(1, 32, 512).bfloat16().to("hpu")
        pos_t, slot_t = [torch.tensor([i], dtype=torch.int32, device="hpu") for i in (pos, slot)]
        lengths = torch.tensor([min(640, 128 + pos)], dtype=torch.int32, device="hpu")
        expected, actual, swa_done, main_done = compiled(query, *inputs, pos_t, slot_t,
                                                         ids.unsqueeze(0).to("hpu"), lengths)
        tag = f"incremental-{offset}-{main_rows}-{generation}"
        exact(tag + "-attention", expected, actual)
        for i, (left, right) in enumerate(zip(refs, caches, strict=True)):
            exact(tag + f"-packed-{i}", left, right)
        exact(tag + "-swa-shadow", unpack_swa(caches[0]), shadow[offset:offset + 512])
        untouched = shadow[512:] if offset == 0 else shadow[:512]
        exact(tag + "-other-layer", torch.full_like(untouched, 17.), untouched)
        exact(tag + "-swa-completion", torch.full_like(swa_done, pos), swa_done)
        if main_rows:
            exact(tag + "-main-shadow", unpack_fp4(caches[1]), main_shadow)
            exact(tag + "-main-completion", torch.full_like(main_done, slot), main_done)


def test_special_values_and_invalid_writes():
    swa = torch.zeros(512, 528, dtype=torch.uint8, device="hpu")
    main = torch.zeros(768, 288, dtype=torch.uint8, device="hpu")
    index = torch.zeros(768, 68, dtype=torch.uint8, device="hpu")
    decoded = torch.zeros(1024, 512, dtype=torch.bfloat16, device="hpu")
    decoded_main = torch.zeros(768, 512, dtype=torch.bfloat16, device="hpu")
    codes = torch.tensor(
        [0, -32768, 1, 127, 128, 16256, -16512, 32640, -128, 32704, 32705, 32767, -63, -1, 32639, -129],
        dtype=torch.int16).view(torch.bfloat16)
    values = codes.repeat_interleave(32).reshape(1, 512)
    reference_swa, reference_main, reference_index = swa.clone(), main.clone(), index.clone()

    def write(value, position):
        old_swa = OPS.custom_deepseek_v41_swa_pack_write_bf16_gaudi2(reference_swa, value, position)
        old_main = OPS.custom_deepseek_v41_fp4_cache_write_bf16_gaudi2(reference_main, reference_index, value,
                                                                       value[:, :128].contiguous(), position)
        a = OPS.custom_deepseek_v41_swa_decoded_write_bf16_gaudi2(swa, value, position, decoded, 512)
        b = OPS.custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2(main, index, value, value[:, :128].contiguous(),
                                                                  position, decoded_main)
        return a, b, old_swa, old_main

    compiled = torch.compile(write, backend="hpu_backend", fullgraph=True, dynamic=False)
    for generation in range(3):
        value = values.roll(generation * 11, dims=1).to("hpu")
        done = compiled(value, torch.tensor([generation], dtype=torch.int32, device="hpu"))
        assert all(bool((t.cpu() == generation).all()) for t in done)
        exact(f"special-{generation}-swa", unpack_swa(swa), decoded[512:])
        exact(f"special-{generation}-main", unpack_fp4(main), decoded_main)
        exact(f"special-{generation}-packed-swa", reference_swa, swa)
        exact(f"special-{generation}-packed-main", reference_main, main)
        exact(f"special-{generation}-packed-index", reference_index, index)
    saved = [x.cpu() for x in (swa, main, index, decoded, decoded_main)]
    for position in (-1, 768):
        done = compiled(values.to("hpu"), torch.tensor([position], dtype=torch.int32, device="hpu"))
        assert all(bool((t.cpu() == -1).all()) for t in done)
        for i, (left, right) in enumerate(zip(saved, (swa, main, index, decoded, decoded_main), strict=True)):
            exact(f"invalid-{position}-{i}", left, right)
