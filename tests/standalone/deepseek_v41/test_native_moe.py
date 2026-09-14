# SPDX-License-Identifier: Apache-2.0
"""Native V4.1 contracts. Device execution requires an explicit module lease."""

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_weights import prepare_q16, prepare_s16  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
DECODE = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2
MOE = torch.ops.custom_op.custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2
HPU = pytest.mark.skipif(os.getenv("DSV41_TEST_HPU") != "1", reason="explicit HPU lease is required")


@HPU
def test_engram_hpu_pinned_staging_reuse_waits_for_consumers():
    from vllm_gaudi.ops.deepseek_v41_host import _TransferSlot
    slot = _TransferSlot(6, 12, 256, "hpu")
    stream = torch.hpu.Stream()
    assert slot.host.is_pinned("hpu")
    for generation in range(4):
        slot.reuse()
        slot.host.fill_(generation + 1)
        with torch.hpu.stream(stream):
            slot.device.copy_(slot.host, non_blocking=True)
            slot.dma_done.record(stream)
        torch.hpu.current_stream().wait_event(slot.dma_done)
        consumed = slot.device.to(torch.int32).sum()
        slot.consumer_done.record(torch.hpu.current_stream())
        slot.inflight = True
        assert consumed.item() == (generation + 1) * slot.host.numel()
    slot.reuse()


@HPU
def test_v41_cache_codecs_compile_and_consume_changing_values():
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4, unpack_swa
    def program(value):
        swa, main, index = pack_swa(value), pack_fp4(value, 16), pack_fp4(value[:, :128], 32)
        return swa, main, index, unpack_swa(swa), unpack_fp4(main), unpack_fp4(index, 128, 32)
    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    for seed in (0, 1):
        torch.manual_seed(seed)
        value = (torch.randn(5, 512) * 3).bfloat16()
        value[0, :32] = 0
        value[1, :32] = -0.
        actual = compiled(value.to("hpu"))
        reference = program(value)
        for index, (result, expected) in enumerate(zip(actual, reference, strict=True)):
            result = result.cpu()
            if not torch.equal(result, expected):
                destination = Path(os.environ["HABANA_LOGS"]).parent / f"codec-mismatch-{seed}-{index}.pt"
                torch.save({"input": value, "actual": result, "expected": expected}, destination)
                mismatch = (result != expected).nonzero()
                pytest.fail(f"Codec output {index} differs at {mismatch.shape[0]} entries; saved {destination}")


@HPU
def test_activation_roundtrip_matches_device_codec_for_encodings_scales_and_shapes():
    from vllm_gaudi.ops.deepseek_v41_math import pack_swa, unpack_swa
    def reference(value):
        return unpack_swa(pack_swa(value), value.shape[-1])
    op = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2
    compiled = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)
    old = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(41)
    # Every BF16 encoding acts as both a value and its group's maximum; this
    # covers all power-of-two scale transitions and nonfinite/zero groups.
    encoding = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    cases = [("all_bf16_group_maxima", encoding.repeat_interleave(32).reshape(256, 8192))]
    for shape in ((1, 5120), (6, 1152), (1, 4096), (1, 1280), (5, 16384)):
        cases.append((str(shape), torch.randn(shape).bfloat16()))
    # Exercise small/subnormal FP8 results under a different group maximum,
    # including halfway values with both even and odd retained mantissas.
    mixed = torch.randn(256, 5120).bfloat16()
    mixed[:, ::32] = torch.exp2(torch.arange(-126, 130, dtype=torch.float32)).clamp_max(2.**127).unsqueeze(1)
    mixed[:, 1::32] = 0.
    mixed[:, 2::32] = -0.
    cases.append(("mixed_scale_boundaries", mixed))
    special = torch.randn(32, 512).bfloat16()
    special[0::4, 0] = float("nan")
    special[1::4, 0] = float("inf")
    special[2::4, 0] = -float("inf")
    special[3::4, :32] = -0.
    cases.append(("mixed_nonfinite_groups", special))
    results = []
    for name, source in cases:
        value = source.to("hpu")
        expected, actual = old(value).cpu(), compiled(value).cpu()
        eager = op(value).cpu()
        equal = actual.view(torch.int16) == expected.view(torch.int16)
        same_algorithm = torch.equal(actual.view(torch.int16), eager.view(torch.int16))
        mismatch = (~equal).nonzero()
        results.append({"case": name, "shape": list(source.shape), "mismatches": len(mismatch),
                        "ordinary_compiled_exact": same_algorithm})
        if len(mismatch):
            torch.save({"input": source, "actual": actual, "expected": expected},
                       Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"roundtrip-{name}.pt")
    import json
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "roundtrip-results.json").write_text(json.dumps(results, indent=2))
    assert all(row["mismatches"] == 0 and row["ordinary_compiled_exact"] for row in results), results


@HPU
def test_selected_packed_kv_preserves_codes_indices_and_attention():
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4, unpack_swa
    gather = torch.ops.custom_op.custom_deepseek_v41_selected_kv_bf16_gaudi2
    attention = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_gaudi2
    def reference(swa, main, ids):
        cache = unpack_swa(swa)
        if main.shape[1] == 288:
            cache = torch.cat((cache, unpack_fp4(main)), 0)
        valid = (ids >= 0) & (ids < cache.shape[0])
        result = cache.index_select(0, ids.flatten().clamp(0, cache.shape[0]-1).long())
        result = result.masked_fill(~valid.flatten().unsqueeze(-1), 0)
        local = torch.arange(ids.numel(), device=ids.device, dtype=torch.int32).unsqueeze(0)
        return result, torch.where(valid, local, -1)
    compiled = torch.compile(gather, backend="hpu_backend", fullgraph=True, dynamic=False)
    old = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    fused_attention = torch.compile(attention, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(4102)
    swa = torch.empty(256, 528, dtype=torch.uint8)
    main = torch.empty(256, 288, dtype=torch.uint8)
    # Every FP8 code meets every UE8M0 code; each FP4 code meets every E4M3 scale.
    swa[:, :512] = torch.arange(512).remainder(256).byte()
    swa[:, 512:] = torch.arange(256).byte().unsqueeze(1)
    main[:, :256] = torch.arange(256).byte()
    main[:, 256:] = torch.arange(256).byte().unsqueeze(1)
    ids = torch.arange(640).unsqueeze(0).int() - 32
    ids[:, :8] = torch.tensor([-1, 0, 511, 255, 256, 256, 512, -100], dtype=torch.int32)
    cases = [("encodings_and_scale_boundaries", swa, main, ids)]
    for seed in range(3):
        torch.manual_seed(seed)
        swa = pack_swa(torch.randn(512, 512).bfloat16())
        main = pack_fp4(torch.randn(256, 512).bfloat16())
        ids = torch.randint(-32, 800, (1, 640), dtype=torch.int32)
        cases.append((f"changing_cache_{seed}", swa, main, ids))
    cases.append(("swa_only_alias", swa, swa, ids))
    records = []
    for name, sw, ma, ix in cases:
        args = tuple(x.to("hpu") for x in (sw, ma, ix))
        expected = old(*args)
        actual = compiled(*args)
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            a, b = a.cpu(), b.cpu()
            same = torch.equal(a.view(torch.int16) if index == 0 else a,
                               b.view(torch.int16) if index == 0 else b)
            if not same:
                torch.save({"inputs": (sw, ma, ix), "actual": a, "expected": b},
                           Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"selected-kv-{name}-{index}.pt")
            records.append({"case": name, "output": index, "equal": same})
        if name.startswith("changing_cache"):
            query = torch.randn(1, 32, 512, dtype=torch.bfloat16, device="hpu")
            sink = torch.zeros(32, dtype=torch.float32, device="hpu")
            scale = torch.tensor([512**-0.5], dtype=torch.float32, device="hpu")
            baseline = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
                query, expected[0], expected[1], sink, scale)[0].cpu()
            output = fused_attention(query, *args, sink, scale).cpu()
            eager = attention(query, *args, sink, scale).cpu()
            records.append({"case": name, "output": "attention", "equal": torch.equal(
                output.view(torch.int16), baseline.view(torch.int16)) and torch.equal(
                output.view(torch.int16), eager.view(torch.int16))})
    import json
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "selected-kv-results.json").write_text(json.dumps(records, indent=2))
    assert all(row["equal"] for row in records), records


@HPU
def test_device_position_bank_changes_rope_across_requests_and_bucket_tails():
    from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank
    from vllm_gaudi.ops.deepseek_v41_math import apply_rope, rotary_table
    bank = PositionBank(512, 6, "hpu")
    positions = torch.empty(6, dtype=torch.int32, device="hpu")
    views = {count: positions[:count] for count in range(1, 7)}
    rotary = rotary_table(64, 512, 10000).to("hpu")
    def program(query, position):
        return apply_rope(query, position, rotary)
    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    for start, count in ((0, 6), (6, 1), (127, 1), (128, 1), (506, 6), (511, 1), (0, 1), (128, 3)):
        query = torch.randn(count, 512, dtype=torch.bfloat16, device="cpu").to("hpu")
        old_positions = torch.arange(start, start + count, dtype=torch.int32, device="cpu").to("hpu")
        expected = compiled(query, old_positions).cpu()
        bank.copy_into(views[count], start)
        actual = compiled(query, views[count]).cpu()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@HPU
def test_bounded_packed_attention_preserves_order_and_changes_live_lengths():
    from vllm_gaudi.ops.deepseek_v41_math import pack_swa, pack_fp4
    torch.manual_seed(4116)
    ordinary = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_gaudi2
    bounded = torch.ops.custom_op.custom_deepseek_v41_packed_attention_bf16_lengths_gaudi2
    compiled = torch.compile(bounded, backend="hpu_backend", fullgraph=True, dynamic=False)
    swa = pack_swa(torch.randn(512, 512).bfloat16()).to("hpu")
    main = pack_fp4(torch.randn(512, 512).bfloat16()).to("hpu")
    query = torch.randn(1, 32, 512, dtype=torch.bfloat16, device="hpu")
    sink = torch.randn(32, dtype=torch.float32, device="hpu")
    scale = torch.tensor([512**-0.5], dtype=torch.float32, device="hpu")
    live_length = torch.empty(1, dtype=torch.int32, device="hpu")
    records = []
    for width in (0, 1, 127, 128, 129, 192, 256, 639, 640, 128):
        ids = torch.randint(0, 1024, (1, 640), dtype=torch.int32)
        ids[:, width:] = -1
        if width > 8:
            ids[:, :8] = torch.tensor([-1, 511, 512, 512, 0, -100, 1024, 31], dtype=torch.int32)
        ids = ids.to("hpu")
        live_length.copy_(torch.tensor([width], dtype=torch.int32))
        args = query, swa, main, ids, sink, scale
        expected = ordinary(*args).cpu().view(torch.int16)
        actual = compiled(*args, live_length).cpu().view(torch.int16)
        eager = bounded(*args, live_length).cpu().view(torch.int16)
        exact = torch.equal(expected, actual) and torch.equal(actual, eager)
        records.append({"width": width, "exact": exact})
    import json
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "bounded-attention-results.json").write_text(json.dumps(records, indent=2))
    assert all(row["exact"] for row in records), records


@HPU
def test_sinkhorn_batch_extension_preserves_all_tokens():
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre
    torch.manual_seed(13)
    residual = torch.randn(5, 4, 128).bfloat16()
    previous = torch.randn(5, 4)
    fn, scale, base = torch.randn(24, 512) * .1, torch.randn(3), torch.randn(24)
    expected = hc_pre(residual, previous, fn, scale, base)
    compiled = torch.compile(hc_pre, backend="hpu_backend", fullgraph=True, dynamic=False)
    actual = compiled(*(value.to("hpu") for value in (residual, previous, fn, scale, base)))
    for result, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.cpu(), target, atol=2e-3, rtol=2e-3)


def prepared(packed, scales, device):
    q16 = np.stack([prepare_q16(matrix)[0] for matrix in packed])
    bits = np.stack([prepare_s16(matrix, (packed.shape[1], packed.shape[2] * 2))[0] for matrix in scales])
    return torch.from_numpy(q16).to(device), torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).to(device)


def decoded_reference(packed, scales):
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)
    raw = torch.from_numpy(packed).long()
    code = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    scale = torch.from_numpy(scales).double()
    factor = torch.exp2(scale - 127)
    factor[scale == 255] = float("nan")
    return (values[code] * factor.repeat_interleave(32, dim=-1)).bfloat16()


@pytest.mark.parametrize("tokens,experts,topk", [(1, 384, 6), (5, 128, 3), (6, 384, 6)])
def test_v41_meta_has_real_tp_shapes(tokens, experts, topk):
    args = [torch.empty(tokens, 5120, dtype=torch.bfloat16, device="meta"),
            torch.empty(tokens, topk, dtype=torch.int32, device="meta"),
            torch.empty(tokens, topk, dtype=torch.float32, device="meta"),
            torch.empty(experts, 18, 163840, dtype=torch.int16, device="meta"),
            torch.empty(experts, 40, 36864, dtype=torch.int16, device="meta"),
            torch.empty(experts, 18, 20480, dtype=torch.bfloat16, device="meta"),
            torch.empty(experts, 40, 4608, dtype=torch.bfloat16, device="meta"),
            torch.empty(128, dtype=torch.bfloat16, device="meta")]
    assert MOE(*args, False).shape == (tokens, 5120)
    assert DECODE(args[1], args[4], args[6], args[7], False).shape == (tokens * topk, 1152, 5120)
    with pytest.raises(RuntimeError, match="S16"):
        MOE(*args[:6], args[6][..., :-128].contiguous(), args[7], False)


@HPU
@pytest.mark.parametrize("normal", [False, True])
def test_all_codes_and_scales_consume_reordered_duplicate_and_invalid_ids(normal):
    torch._dynamo.reset()
    packed = np.tile((np.arange(16, dtype=np.uint8) * 17), (1, 256, 4))
    scales = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 4, axis=2)
    if normal:
        scales = np.clip(scales, 2, 254)
    reference = decoded_reference(packed, scales)[0].T.contiguous()
    q16, s16 = prepared(packed, scales, "hpu")
    lookup = mxfp4_bf16_lut(torch.device("hpu"))
    compiled = torch.compile(DECODE, backend="hpu_backend", fullgraph=True, dynamic=False)
    for values in ([[0, -1, 1], [0, 0, 0]], [[1, 0, 0], [-1, 0, 1]]):
        ids = torch.tensor(values, dtype=torch.int32, device="hpu")
        actual = compiled(ids, q16, s16, lookup, normal).cpu()
        for index, expert in enumerate(sum(values, [])):
            expected = reference if expert == 0 else torch.zeros_like(reference)
            assert torch.equal(torch.isnan(actual[index]), torch.isnan(expected))
            mask = ~torch.isnan(expected)
            assert torch.equal(actual[index].view(torch.int16)[mask], expected.view(torch.int16)[mask])


@HPU
@pytest.mark.parametrize("tokens,topk", [(1, 6), (5, 3)])
def test_full_moe_graph_consumes_token_routing_and_v41_clamp(tokens, topk):
    torch._dynamo.reset()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    hidden, intermediate, experts = 5120, 1152, 2
    q13 = rng.integers(0, 256, (experts, intermediate * 2, hidden // 2), dtype=np.uint8)
    q2 = rng.integers(0, 256, (experts, hidden, intermediate // 2), dtype=np.uint8)
    s13 = np.full((experts, intermediate * 2, hidden // 32), 119, dtype=np.uint8)
    s2 = np.full((experts, hidden, intermediate // 32), 119, dtype=np.uint8)
    w13, w2 = decoded_reference(q13, s13), decoded_reference(q2, s2)
    prepared13, scale13 = prepared(q13, s13, "hpu")
    prepared2, scale2 = prepared(q2, s2, "hpu")
    lookup = mxfp4_bf16_lut(torch.device("hpu"))
    compiled = torch.compile(MOE, backend="hpu_backend", fullgraph=True, dynamic=False)
    previous = None
    for step in range(2):
        x = (torch.randn(tokens, hidden) * (1 if step == 0 else 8)).bfloat16()
        ids = (torch.arange(tokens * topk).reshape(tokens, topk) + step).remainder(experts).int()
        router = torch.rand(tokens, topk)
        router = router / router.sum(-1, keepdim=True) * 1.5
        actual = compiled(x.to("hpu"), ids.to("hpu"), router.to("hpu"), prepared13, prepared2,
                          scale13, scale2, lookup, True).cpu()
        expected = torch.zeros(tokens, hidden, dtype=torch.float32)
        for token in range(tokens):
            for rank in range(topk):
                expert = ids[token, rank]
                first = (x[token].float() @ w13[expert].float().T).bfloat16().float()
                gate, up = first.chunk(2)
                middle = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10) * router[token, rank]).bfloat16()
                expected[token] += (middle.float() @ w2[expert].float().T).bfloat16().float()
        # Different CPU/MME reduction orders are not an addressing failure.
        # Final model qualification still uses the frozen production reference.
        torch.testing.assert_close(actual.float(), expected.bfloat16().float(), atol=.03125, rtol=.03)
        assert torch.isfinite(actual).all()
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual
