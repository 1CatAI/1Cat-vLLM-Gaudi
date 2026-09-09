# SPDX-License-Identifier: Apache-2.0
"""Contracts for the opt-in indexed decoder and compound BF16 MME op.

HPU tests require DSV4_TEST_HPU=1 and explicitly selected free modules.
"""
import os
from types import SimpleNamespace

import pytest

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from vllm_gaudi.ops.deepseek_v4_mxfp4 import (  # noqa: E402
    PreparedMxfp4Weight,
    mxfp4_bf16_lut,
    normal_e8m0_scales,
    prepare_mxfp4_q16,
    prepare_mxfp4_s16,
    restore_mxfp4_scale_u8,
    restore_mxfp4_u8,
)

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
DEQUANT = torch.ops.custom_op.custom_deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2
LINEAR = torch.ops.custom_op.custom_deepseek_v4_mxfp4_indexed_linear_bf16_gaudi2
MOE = torch.ops.custom_op.custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2
PREPARED_DEQUANT = torch.ops.custom_op.custom_deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2
PREPARED_LINEAR = torch.ops.custom_op.custom_deepseek_v4_mxfp4_prepared_linear_bf16_gaudi2
PREPARED_EXPERTS = torch.ops.custom_op.custom_deepseek_v4_mxfp4_prepared_experts_bf16_gaudi2
PREPARED_MOE = torch.ops.custom_op.custom_deepseek_v4_mxfp4_prepared_moe_bf16_gaudi2
WEIGHTED_SUM = torch.ops.custom_op.custom_deepseek_v4_weighted_sum_bf16_gaudi2
HPU = pytest.mark.skipif(os.environ.get("DSV4_TEST_HPU") != "1", reason="explicit HPU selection required")


@pytest.fixture(autouse=True)
def independent_compile_contracts():
    # Each test owns its compiled functions. Keep the multiple shape/boolean
    # contracts from sharing torch.compile(OpOverloadPacket)'s wrapper cache;
    # dynamic-ID replay remains inside each test and is not reset.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def decode_reference(ids, packed, scales):
    """Independent numeric E2M1/E8M0 reference; no bit construction."""
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)
    result = []
    for expert in ids.flatten().tolist():
        if expert < 0 or expert >= packed.shape[0]:
            result.append(torch.zeros(packed.shape[1], packed.shape[2] * 2, dtype=torch.bfloat16))
            continue
        weight = packed[expert].long()
        codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
        exponents = scales[expert].double()
        factor = torch.exp2(exponents - 127)
        factor[exponents == 255] = float("nan")
        result.append((table[codes] * factor.repeat_interleave(32, dim=-1)).bfloat16())
    return torch.stack(result)


def assert_decode_equal(actual, expected):
    actual = actual.cpu()
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    finite_or_inf = ~torch.isnan(expected)
    # Includes negative zero, subnormals and signed infinities.
    actual_bits = actual.view(torch.int16)
    expected_bits = expected.view(torch.int16)
    mismatch = finite_or_inf & (actual_bits != expected_bits)
    if mismatch.any():
        locations = torch.nonzero(mismatch)[:16]
        details = [(location.tolist(), int(actual_bits[tuple(location)]), int(expected_bits[tuple(location)]))
                   for location in locations]
        pytest.fail(f"BF16 bit mismatch count={int(mismatch.sum())}, first={details}")


def meta_inputs():
    return [torch.empty(1, 4096, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 6, device="meta", dtype=torch.int32),
            torch.empty(1, 6, device="meta", dtype=torch.bfloat16),
            torch.empty(256, 2048, 2048, device="meta", dtype=torch.uint8),
            torch.empty(256, 4096, 512, device="meta", dtype=torch.uint8),
            torch.empty(256, 2048, 128, device="meta", dtype=torch.uint8),
            torch.empty(256, 4096, 32, device="meta", dtype=torch.uint8)]


def prepared_meta_inputs():
    return [torch.empty(1, 4096, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 6, device="meta", dtype=torch.int32),
            torch.empty(1, 6, device="meta", dtype=torch.bfloat16),
            torch.empty(256, 16, 131072, device="meta", dtype=torch.int16),
            torch.empty(256, 32, 32768, device="meta", dtype=torch.int16),
            torch.empty(256, 16, 16384, device="meta", dtype=torch.bfloat16),
            torch.empty(256, 32, 4096, device="meta", dtype=torch.bfloat16),
            torch.empty(128, device="meta", dtype=torch.bfloat16),
            torch.empty(1, 6, device="meta", dtype=torch.int32),
            torch.empty(1, 6, device="meta", dtype=torch.int32)]


def test_meta_contract():
    args = meta_inputs()
    output = MOE(*args)
    assert output.shape == (1, 4096) and output.dtype == torch.bfloat16
    assert output.device.type == "meta"
    assert DEQUANT(args[1], args[3], args[5]).shape == (6, 2048, 4096)
    assert LINEAR(args[0], args[1], args[3], args[5]).shape == (6, 2048)
    assert MOE(*args, True).shape == output.shape


def test_prepared_meta_contract():
    args = prepared_meta_inputs()
    expert_outputs = PREPARED_EXPERTS(args[0], args[1], *args[3:8])
    assert expert_outputs.shape == (6, 4096) and expert_outputs.dtype == torch.bfloat16
    assert expert_outputs.device.type == "meta"
    output = WEIGHTED_SUM(expert_outputs, args[2], args[8], args[9])
    assert output.shape == (1, 4096) and output.dtype == torch.bfloat16
    assert output.device.type == "meta"
    fused_output = PREPARED_MOE(*args[:8])
    assert fused_output.shape == (1, 4096) and fused_output.dtype == torch.bfloat16
    assert fused_output.device.type == "meta"
    assert PREPARED_DEQUANT(args[1], args[3], args[5], args[7]).shape == (6, 4096, 2048)
    assert PREPARED_LINEAR(args[0], args[1], args[3], args[5], args[7]).shape == (6, 2048)
    assert PREPARED_EXPERTS(args[0], args[1], *args[3:8], True).shape == expert_outputs.shape


def test_prepared_compat_meta_contract():
    import vllm_gaudi.extension.ops  # noqa: F401

    args = prepared_meta_inputs()
    output = torch.ops.vllm_gaudi.deepseek_v4_mxfp4_prepared_compat(*args[:7])
    assert output.shape == (1, 4096)
    assert output.dtype == torch.bfloat16
    assert output.device.type == "meta"


def test_prepared_layout_and_scale_roundtrip():
    values = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 256)
    packed = values.expand(1, 256, 256).contiguous()
    q16 = prepare_mxfp4_q16(packed)
    assert q16.shape == (1, 2, 16384) and q16.dtype == torch.int16
    assert torch.equal(restore_mxfp4_u8(q16), packed)
    scales = torch.arange(256, dtype=torch.uint8).reshape(1, 256, 1).expand(1, 256, 128).contiguous()
    s16 = prepare_mxfp4_s16(scales)
    assert s16.shape == (1, 2, 16384) and s16.dtype == torch.bfloat16
    assert torch.equal(restore_mxfp4_scale_u8(s16), scales)
    lookup = mxfp4_bf16_lut("cpu")
    expected = torch.tensor([0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
                             0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC], dtype=torch.uint8)
    for offset in range(0, 128, 16):
        start = offset * 2
        assert torch.equal(lookup.view(torch.uint8)[start:start + 16], expected)
        assert torch.equal(lookup.view(torch.uint8)[start + 16:start + 32], expected)


def test_prepared_route_preserves_stock_accumulation_order():
    """Protect a real checkpoint cancellation case from reduce reassociation."""
    from vllm_gaudi.extension.ops import _prepared_mxfp4_weighted_sum_fp32

    expert_outputs = torch.tensor([
        0.0016021728515625,
        0.001800537109375,
        -1.8477439880371094e-05,
        0.00921630859375,
        -0.007781982421875,
        -0.00102996826171875,
    ], dtype=torch.bfloat16).reshape(6, 1)
    router_weights = torch.tensor([
        0.259765625,
        0.03759765625,
        0.2001953125,
        0.1826171875,
        0.271484375,
        0.049072265625,
    ], dtype=torch.bfloat16).reshape(1, 6)

    output = _prepared_mxfp4_weighted_sum_fp32(expert_outputs, router_weights)
    assert output.shape == (1, 1)
    assert int(output.view(torch.int16).item()) == 12896


@pytest.mark.parametrize("index,dtype", [(0, torch.float16), (1, torch.int64), (2, torch.float32),
                                         (3, torch.int8), (5, torch.float32)])
def test_reject_dtype(index, dtype):
    args = meta_inputs()
    args[index] = args[index].to(dtype)
    with pytest.raises(RuntimeError, match="dtype/device"):
        MOE(*args)


def test_reject_batch_layout_scale_and_autograd():
    for index, value in [(0, torch.empty(2, 4096, dtype=torch.bfloat16, device="meta")),
                         (0, torch.empty(4096, 1, dtype=torch.bfloat16, device="meta").t()),
                         (5, torch.empty(256, 2048, 64, dtype=torch.uint8, device="meta")),
                         (0, torch.empty(1, 4096, dtype=torch.bfloat16, device="meta", requires_grad=True))]:
        args = meta_inputs()
        # A transposed singleton dimension is still contiguous: use a strided slice.
        if index == 0 and not value.requires_grad and value.shape == (1, 4096):
            value = torch.empty(1, 8192, dtype=torch.bfloat16, device="meta")[:, ::2]
        args[index] = value
        with pytest.raises(RuntimeError):
            MOE(*args)


def test_candidate_disabled_by_default(monkeypatch):
    from vllm_gaudi import envs
    monkeypatch.delenv("VLLM_HPU_DSV4_MXFP4_INDEXED_MME", raising=False)
    monkeypatch.delenv("VLLM_HPU_DSV4_MXFP4_PREPARED_MME", raising=False)
    assert not envs.VLLM_HPU_DSV4_MXFP4_INDEXED_MME
    assert not envs.VLLM_HPU_DSV4_MXFP4_PREPARED_MME


@pytest.mark.parametrize("enabled,tp,gaudi2,tokens,selected", [
    (True, 2, True, 1, True),
    (False, 2, True, 1, False),
    (True, 1, True, 1, False),
    (True, 4, True, 1, False),
    (True, 2, False, 1, False),
    (True, 2, True, 2, False),
])
def test_model_dispatch_scope(monkeypatch, enabled, tp, gaudi2, tokens, selected):
    import vllm_gaudi.extension.ops as ops

    monkeypatch.setenv("VLLM_HPU_DSV4_MXFP4_INDEXED_MME", str(int(enabled)))
    monkeypatch.setenv("VLLM_HPU_DSV4_MXFP4_PREPARED_MME", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_MXFP4_INDEXED", "0")
    monkeypatch.setenv("VLLM_HPU_MXFP4_DECODE_GATHER", "0")
    monkeypatch.setattr(ops, "is_hpu_gaudi2", gaudi2)
    monkeypatch.setattr(ops, "get_config", lambda: SimpleNamespace(moe_chunk=[], moe_token_boundary=[]))
    layer = ops.VllmMixtureOfExpertsOpMXFP4(256, 256, 0, 255, tensor_parallel_size=tp)
    args = meta_inputs()
    if tokens != 1:
        args[:3] = [t.expand(tokens, -1).contiguous() for t in args[:3]]
    layer.set_stacked_weights(*args[3:])
    for index in range(256):
        layer.w13_list[index].set_weight(args[3][index])
        layer.w2_list[index].set_weight(args[4][index])
        layer.w13_list[index].set_scale(args[5][index])
        layer.w2_list[index].set_scale(args[6][index])
    calls = []

    def candidate(*inputs):
        calls.append("candidate")
        return MOE(*inputs)

    def fallback(*inputs):
        calls.append("fallback")
        return torch.empty_like(inputs[0])

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2", candidate)
    layer._compiled_forward = fallback
    output = layer(*args[:3])
    assert calls == ["candidate" if selected else "fallback"]
    assert output.shape == (tokens, 4096) and output.dtype == torch.bfloat16


@pytest.mark.parametrize("enabled,tp,gaudi2,tokens,selected", [
    (True, 2, True, 1, True),
    (False, 2, True, 1, False),
    (True, 1, True, 1, False),
    (True, 4, True, 1, False),
    (True, 2, False, 1, False),
    (True, 2, True, 2, False),
])
def test_prepared_model_dispatch_scope(monkeypatch, enabled, tp, gaudi2, tokens, selected):
    import vllm_gaudi.extension.ops as ops

    monkeypatch.setenv("VLLM_HPU_DSV4_MXFP4_PREPARED_MME", str(int(enabled)))
    monkeypatch.setenv("VLLM_HPU_DSV4_MXFP4_INDEXED_MME", "0")
    monkeypatch.setenv("VLLM_HPU_DSV4_TPC_MXFP4_INDEXED", "0")
    monkeypatch.setenv("VLLM_HPU_MXFP4_DECODE_GATHER", "0")
    monkeypatch.setattr(ops, "is_hpu_gaudi2", gaudi2)
    monkeypatch.setattr(ops, "get_config", lambda: SimpleNamespace(moe_chunk=[], moe_token_boundary=[]))
    layer = ops.VllmMixtureOfExpertsOpMXFP4(256, 256, 0, 255, tensor_parallel_size=tp)
    args = prepared_meta_inputs()
    w13 = PreparedMxfp4Weight(
        args[3], args[5], (256, 2048, 2048), (256, 2048, 128), True, layer._weight_generation
    )
    w2 = PreparedMxfp4Weight(
        args[4], args[6], (256, 4096, 512), (256, 4096, 32), True, layer._weight_generation
    )
    layer.set_prepared_weights(w13, w2, args[7])
    if tokens != 1:
        args[:3] = [value.expand(tokens, -1).contiguous() for value in args[:3]]
    calls = []

    def candidate(*inputs):
        calls.append("candidate")
        return torch.empty(1, 4096, dtype=inputs[0].dtype, device=inputs[0].device)

    def fallback(*inputs):
        calls.append("fallback")
        return torch.empty_like(inputs[0])

    monkeypatch.setattr(
        torch.ops.custom_op,
        "custom_deepseek_v4_mxfp4_prepared_moe_bf16_gaudi2",
        candidate,
    )
    layer._compiled_forward = fallback
    output = layer(*args[:3])
    assert calls == (["candidate"] if selected else ["fallback"])
    assert output.shape == (tokens, 4096) and output.dtype == torch.bfloat16


@HPU
@pytest.mark.parametrize("normal", [False, True])
def test_all_codes_scales_dynamic_ids_and_replay(normal):
    assert os.environ.get("HABANA_VISIBLE_MODULES") and os.environ.get("HLS_MODULE_ID")
    codes = torch.arange(2, 255) if normal else torch.arange(256)
    rows = len(codes)
    packed = ((torch.arange(256).view(1, 1, 256) + torch.arange(8).view(8, 1, 1) * 7)
              % 256).expand(8, rows, 256).byte().contiguous()
    scales = codes.view(1, rows, 1).expand(8, rows, 16).byte().contiguous()
    assert normal_e8m0_scales(scales) == normal
    weights_hpu, scales_hpu = packed.to("hpu"), scales.to("hpu")
    ids_hpu = torch.zeros(1, 6, device="hpu", dtype=torch.int32)
    fn = torch.compile(DEQUANT, backend="hpu_backend", fullgraph=True, dynamic=False)
    sequences = [[7, 2, 0, 6, 1, 5], [0, 0, 7, 7, 4, 4], [-1, 8, 0, 2, 3, 7], [7, 2, 0, 6, 1, 5]]
    for sequence in sequences:
        ids = torch.tensor([sequence], dtype=torch.int32)
        ids_hpu.copy_(ids)
        actual = fn(ids_hpu, weights_hpu, scales_hpu, normal)
        assert_decode_equal(actual, decode_reference(ids, packed, scales))
    assert torch.equal(weights_hpu.cpu(), packed)
    assert torch.equal(scales_hpu.cpu(), scales)


@HPU
def test_native_prefill_outer_compile_retains_compatible_boundary(monkeypatch):
    """An outer model compile must not reintroduce the unsupported FP4 type."""
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    import vllm_gaudi.extension.ops as ops

    # The HPU FakeTensor wrapper also initializes the selected device.
    assert os.environ.get("HABANA_VISIBLE_MODULES") and os.environ.get("HLS_MODULE_ID")
    monkeypatch.setenv("VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH", "1")
    monkeypatch.setenv("VLLM_HPU_DSV4_MXFP4_PREPARED_MME", "1")
    monkeypatch.setattr(ops, "get_config", lambda: SimpleNamespace(moe_chunk=[], moe_token_boundary=[]))
    mode = FakeTensorMode()
    with mode:
        args = [FakeTensor(mode, value, torch.device("hpu")) for value in prepared_meta_inputs()]
        layer = ops.VllmMixtureOfExpertsOpMXFP4(256, 256, 0, 255, tensor_parallel_size=2)
        layer.set_prepared_weights(
            PreparedMxfp4Weight(args[3], args[5], (256, 2048, 2048), (256, 2048, 128), False, 0),
            PreparedMxfp4Weight(args[4], args[6], (256, 4096, 512), (256, 4096, 32), False, 0), args[7])
        incoming = [value.expand(3, -1).contiguous() for value in args[:3]]
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        output = torch.compile(layer, backend=backend, fullgraph=True)(*incoming)
    assert output.shape == (3, 4096)
    calls = [str(node.target) for graph in graphs for node in graph.graph.nodes if node.op == "call_function"]
    assert any("deepseek_v4_mxfp4_prepared_compat" in target for target in calls)
    assert not any("mixture_of_experts" in target for target in calls)


@HPU
@pytest.mark.parametrize("normal", [False, True])
def test_prepared_all_codes_scales_dynamic_ids_and_replay(normal):
    assert os.environ.get("HABANA_VISIBLE_MODULES") and os.environ.get("HLS_MODULE_ID")
    scale_codes = torch.cat((torch.arange(2, 255), torch.tensor([2, 3, 254]))) if normal else torch.arange(256)
    rows = len(scale_codes)
    packed = ((torch.arange(256).view(1, 1, 256) + torch.arange(8).view(8, 1, 1) * 7)
              % 256).expand(8, rows, 256).byte().contiguous()
    scales = scale_codes.view(1, rows, 1).expand(8, rows, 16).byte().contiguous()
    q16 = prepare_mxfp4_q16(packed).to("hpu")
    s16 = prepare_mxfp4_s16(scales).to("hpu")
    lookup = mxfp4_bf16_lut("hpu")
    ids_hpu = torch.zeros(1, 6, device="hpu", dtype=torch.int32)
    fn = torch.compile(PREPARED_DEQUANT, backend="hpu_backend", fullgraph=True, dynamic=False)
    sequences = [[7, 2, 0, 6, 1, 5], [0, 0, 7, 7, 4, 4], [-1, 8, 0, 2, 3, 7], [7, 2, 0, 6, 1, 5]]
    for sequence in sequences:
        ids = torch.tensor([sequence], dtype=torch.int32)
        ids_hpu.copy_(ids)
        actual = fn(ids_hpu, q16, s16, lookup, normal).transpose(1, 2).contiguous()
        assert_decode_equal(actual, decode_reference(ids, packed, scales))
    assert torch.equal(restore_mxfp4_u8(q16.cpu()), packed)
    assert torch.equal(restore_mxfp4_scale_u8(s16.cpu()), scales)


@HPU
@pytest.mark.parametrize("batch", [1, 6])
@pytest.mark.parametrize("normal", [False, True])
def test_indexed_mme_linear(batch, normal):
    torch.manual_seed(20260907)
    packed = torch.randint(0, 256, (8, 256, 256), dtype=torch.uint8)
    scales = torch.randint(117, 127, (8, 256, 16), dtype=torch.uint8)
    ids = torch.tensor([[7, 0, 4, 1, 3, 6]], dtype=torch.int32)
    x = torch.randn(batch, 512).bfloat16()
    expected = torch.matmul(x.float().view(batch, 1, 512),
                            decode_reference(ids, packed, scales).float().transpose(1, 2)).squeeze(1).bfloat16()
    inputs = [t.to("hpu") for t in (x, ids, packed, scales)]
    fn = torch.compile(LINEAR, backend="hpu_backend", fullgraph=True, dynamic=False)
    output = fn(*inputs, normal).cpu()
    torch.testing.assert_close(output, expected, rtol=0.008, atol=2e-4)
    changed_ids = ids.flip(1).contiguous()
    inputs[1].copy_(changed_ids)
    expected = torch.matmul(x.float().view(batch, 1, 512),
                            decode_reference(changed_ids, packed, scales).float().transpose(1, 2)).squeeze(1).bfloat16()
    torch.testing.assert_close(fn(*inputs, normal).cpu(), expected, rtol=0.008, atol=2e-4)


@HPU
@pytest.mark.parametrize("batch", [1, 6])
@pytest.mark.parametrize("normal", [False, True])
def test_prepared_linear_matches_indexed(batch, normal):
    torch.manual_seed(20260907)
    packed = torch.randint(0, 256, (8, 256, 256), dtype=torch.uint8)
    scales = torch.randint(117, 127, (8, 256, 16), dtype=torch.uint8)
    ids = torch.tensor([[7, 0, 4, 1, 3, 6]], dtype=torch.int32)
    x = torch.randn(batch, 512).bfloat16()
    original_inputs = [value.to("hpu") for value in (x, ids, packed, scales)]
    prepared_inputs = [original_inputs[0], original_inputs[1], prepare_mxfp4_q16(packed).to("hpu"),
                       prepare_mxfp4_s16(scales).to("hpu"), mxfp4_bf16_lut("hpu")]
    indexed = torch.compile(LINEAR, backend="hpu_backend", fullgraph=True, dynamic=False)
    prepared = torch.compile(PREPARED_LINEAR, backend="hpu_backend", fullgraph=True, dynamic=False)
    expected = indexed(*original_inputs, normal).cpu()
    actual = prepared(*prepared_inputs, normal).cpu()
    assert torch.equal(actual, expected)


@HPU
@pytest.mark.parametrize("normal", [False, True])
def test_loaded_checkpoint_decoding(normal):
    path = os.environ.get("DSV4_MXFP4_SAMPLE")
    if not path:
        pytest.skip("set DSV4_MXFP4_SAMPLE to a saved loaded-weight sample")
    sample = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    ids = torch.tensor([[3, 0, 2, 1, 3, 0]], dtype=torch.int32)
    fn = torch.compile(DEQUANT, backend="hpu_backend", fullgraph=True, dynamic=False)
    for key in ("w13", "w2"):
        packed, scales = sample[key], sample[key + "_scale"]
        assert packed.shape[0] == 4
        if normal:
            assert normal_e8m0_scales(scales)
        actual = fn(ids.to("hpu"), packed.to("hpu"), scales.to("hpu"), normal)
        assert_decode_equal(actual, decode_reference(ids, packed, scales))


def test_scale_specialization_uses_complete_range():
    assert not normal_e8m0_scales()
    assert not normal_e8m0_scales(torch.empty(0, dtype=torch.uint8))
    assert not normal_e8m0_scales(torch.ones(2))
    normal = torch.tensor([[2, 127], [253, 254]], dtype=torch.uint8)
    assert normal_e8m0_scales(normal, normal.clone())
    for code in (0, 1, 255):
        changed = normal.clone()
        changed[-1, -1] = code
        assert not normal_e8m0_scales(normal, changed)


@HPU
def test_load_time_scale_check_on_hpu():
    values = torch.tensor([2, 127, 253, 254], dtype=torch.uint8, device="hpu")
    assert normal_e8m0_scales(values)
    values[2] = 1
    assert not normal_e8m0_scales(values)
