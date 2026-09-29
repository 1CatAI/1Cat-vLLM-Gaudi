# SPDX-License-Identifier: Apache-2.0
"""Independent group scales and partial-vector tails on a leased Gaudi2."""

import os
from types import SimpleNamespace

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Vector codec tests require an explicit HPU lease", allow_module_level=True)

rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402

prepare_environment()

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

torch.hpu.set_device(rank)
torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
OLD = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2
WIDE = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def assert_same_bits(actual, expected):
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


def test_every_bf16_encoding_and_nonfinite_group_maxima():
    compiled = torch.compile(WIDE, backend="hpu_backend", fullgraph=True, dynamic=False)
    bits = torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16)
    generator = torch.Generator().manual_seed(4132)
    cases = (
        bits.repeat_interleave(32).reshape(256, 8192),
        bits[torch.randperm(bits.numel(), generator=generator)].reshape(512, 128),
    )
    for source in cases:
        value = source.to("hpu")
        expected = OLD(value)
        assert_same_bits(WIDE(value), expected)
        assert_same_bits(compiled(value), expected)


@pytest.mark.parametrize("width", [32, 64, 96, 128, 160, 1280, 5120])
def test_tail_groups_are_independent_and_changed_inputs_are_consumed(width):
    compiled = torch.compile(WIDE, backend="hpu_backend", fullgraph=True, dynamic=False)
    generator = torch.Generator().manual_seed(width)
    scale = torch.exp2((torch.arange(width // 32) % 40 - 20).float())
    device = torch.empty(7, width, dtype=torch.bfloat16, device="hpu")
    for generation in range(3):
        source = torch.randn(7, width // 32, 32, generator=generator) * scale[None, :, None]
        source[generation].zero_()
        source = source.reshape(7, width).bfloat16()
        device.copy_(source)
        expected = OLD(device)
        assert_same_bits(WIDE(device), expected)
        assert_same_bits(compiled(device), expected)


@pytest.mark.parametrize("tokens", [1, 8, 127])
def test_woa_emission_keeps_bf16_rounding_and_channel_order(tokens):
    reference = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_gaudi2
    candidate = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_wide_gaudi2
    compiled = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    generator = torch.Generator().manual_seed(4192)
    # Finite E4M3 codes with distinct channels/groups expose transpose and scale
    # indexing errors without relying on a checkpoint fixture in CI.
    codes = (torch.arange(4 * 4096 * 1024, dtype=torch.int32) % 112).byte()
    weight = codes.reshape(4, 4096, 1024).to("hpu").view(torch.float8_e4m3fn)
    scales = torch.exp2((torch.arange(4 * 1024) % 9 - 4).float()).reshape(4, 1, 1024).to("hpu")
    for generation in range(2):
        source = torch.randn(tokens, 4, 4096, generator=generator).bfloat16()
        source[:, generation].zero_()
        value = source.to("hpu")
        expected = reference(value, weight, scales)
        assert_same_bits(candidate(value, weight, scales), expected)
        assert_same_bits(compiled(value, weight, scales), expected)


@pytest.mark.parametrize("shape,dtype", [((1, 31), torch.bfloat16), ((8193, 32), torch.bfloat16),
                                         ((1, 128), torch.float32)])
def test_rejects_invalid_codec_contract(shape, dtype):
    with pytest.raises(RuntimeError, match="contiguous BF16"):
        WIDE(torch.empty(shape, dtype=dtype, device="meta"))


def test_selected_vector_dispatch_preserves_decode_with_a_diagnostic_disable(monkeypatch):
    from vllm_gaudi import envs
    from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention
    from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_QUANT_ROUNDTRIP", "1")
    assert envs.VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT
    called = []

    def recorder(name):

        def invoke(value, *args):
            called.append(name)
            return value

        return invoke

    for name, kind in (("quant_roundtrip_bf16", "old"), ("quant_roundtrip_wide_bf16", "wide"),
                       ("woa_fp8_roundtrip", "old"), ("woa_fp8_roundtrip_wide", "wide")):
        monkeypatch.setattr(torch.ops.custom_op, f"custom_deepseek_v41_{name}_gaudi2", recorder(kind))
    owner = SimpleNamespace(woa_fp8=True,
                            woa_output_roundtrip=True,
                            weights=SimpleNamespace(wo_a=SimpleNamespace(weight=None, channel_scale=None)))
    for tokens, expected in ((1, "old"), (6, "old"), (7, "old"), (8, "old"), (16, "wide"), (32, "wide")):
        quantize_activation(torch.empty(tokens, 160, dtype=torch.bfloat16, device="hpu"))
        assert called.pop() == expected
        for cls in (CSA2Attention, PagedCSA2Attention):
            cls.project_output(owner, torch.empty(tokens, 4, 4096, dtype=torch.bfloat16, device="hpu"))
            assert called.pop() == expected
    quantize_activation(torch.empty(8, 4, 160, dtype=torch.bfloat16, device="hpu"))
    assert called.pop() == "old"
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT", "0")
    quantize_activation(torch.empty(8, 160, dtype=torch.bfloat16, device="hpu"))
    assert called.pop() == "old"
