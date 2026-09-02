# SPDX-License-Identifier: Apache-2.0

import shutil

import pytest
import torch

from vllm_gaudi.ops.triton_gaudi import runtime


@pytest.fixture(autouse=True)
def reset_runtime(monkeypatch: pytest.MonkeyPatch):
    runtime._reset_for_tests()
    monkeypatch.delenv("VLLM_HPU_TRITON_MODE", raising=False)
    yield
    runtime._reset_for_tests()


def test_off_mode_preserves_vendor_add():
    lhs = torch.arange(8, dtype=torch.float32)
    rhs = torch.ones_like(lhs)

    result = runtime.vector_add(lhs, rhs)

    torch.testing.assert_close(result, lhs + rhs)
    assert runtime.diagnostics()["counters"] == {"vendor.off": 1}


def test_hybrid_mode_records_non_hpu_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")
    lhs = torch.arange(8, dtype=torch.bfloat16)

    result = runtime.vector_add(lhs, lhs)

    torch.testing.assert_close(result, lhs + lhs)
    assert runtime.diagnostics()["counters"] == {"fallback.non_hpu_tensor": 1}


def test_strict_mode_rejects_non_hpu_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")
    lhs = torch.arange(8, dtype=torch.float32)

    with pytest.raises(runtime.FastPathUnavailable, match="non_hpu_tensor"):
        runtime.vector_add(lhs, lhs)


def test_invalid_mode_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "sometimes")

    with pytest.raises(ValueError, match="must be one of"):
        runtime.diagnostics()


def test_off_mode_leaves_fused_add_rms_norm_on_vendor_path():
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)
    residual = torch.ones_like(hidden)
    weight = torch.ones(8, dtype=torch.bfloat16)

    result = runtime.fused_add_rms_norm(hidden, residual, weight, 1.0e-6)

    assert result is None
    assert runtime.diagnostics()["counters"] == {"vendor.fused_add_rms_norm.off": 1}


def test_hybrid_fused_add_rms_norm_records_non_hpu_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)
    residual = torch.ones_like(hidden)
    weight = torch.ones(8, dtype=torch.bfloat16)

    result = runtime.fused_add_rms_norm(hidden, residual, weight, 1.0e-6)

    assert result is None
    assert runtime.diagnostics()["counters"] == {"fallback.fused_add_rms_norm.non_hpu_tensor": 1}


def test_strict_fused_add_rms_norm_rejects_non_hpu_tensor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)

    with pytest.raises(runtime.FastPathUnavailable, match="fused_add_rms_norm: non_hpu_tensor"):
        runtime.fused_add_rms_norm(hidden, hidden, torch.ones(8, dtype=torch.bfloat16), 1.0e-6)


def test_compile_off_mode_does_not_enable_fast_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)

    result = runtime.fused_add_rms_norm(hidden, hidden, torch.ones(8, dtype=torch.bfloat16), 1.0e-6)

    assert result is None
    assert runtime.diagnostics()["counters"] == {}


def test_compile_mode_requires_early_preparation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)

    with pytest.raises(runtime.FastPathUnavailable, match="must be prepared before torch.compile"):
        runtime.fused_add_rms_norm(hidden, hidden, torch.ones(8, dtype=torch.bfloat16), 1.0e-6)


def test_compile_hybrid_mode_keeps_ungated_kernels_on_vendor_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(runtime, "_prepared", True)
    monkeypatch.setattr(runtime, "_available", True)
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)

    rms_result = runtime.fused_add_rms_norm(
        hidden,
        hidden,
        torch.ones(8, dtype=torch.bfloat16),
        1.0e-6,
    )
    silu_result = runtime.silu_and_mul(hidden)
    gdn_result = runtime.gdn_decode_packed(*_cpu_gdn_inputs())

    assert rms_result is None
    assert silu_result is None
    assert gdn_result is None
    assert runtime.diagnostics()["counters"] == {}


def test_compile_hybrid_gdn_keeps_large_batch_on_vendor_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(runtime, "_prepared", True)
    monkeypatch.setattr(runtime, "_available", True)
    monkeypatch.setattr(
        runtime,
        "_gdn_decode_packed_rejection_reason",
        lambda *args: None,
    )
    result = runtime.gdn_decode_packed(*_cpu_gdn_inputs(batch=8))

    assert result is None


def test_eager_hybrid_gdn_keeps_candidate_on_vendor_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")
    monkeypatch.setattr(
        runtime,
        "_gdn_decode_packed_rejection_reason",
        lambda *args: None,
    )

    assert runtime.gdn_decode_packed(*_cpu_gdn_inputs(batch=8)) is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_packed.performance_gate": 1,
    }


def test_off_mode_leaves_silu_and_mul_on_vendor_path():
    input_tensor = torch.zeros(2, 16, dtype=torch.bfloat16)

    result = runtime.silu_and_mul(input_tensor)

    assert result is None
    assert runtime.diagnostics()["counters"] == {"vendor.silu_and_mul.off": 1}


def test_hybrid_silu_and_mul_records_non_hpu_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")
    input_tensor = torch.zeros(2, 16, dtype=torch.bfloat16)

    result = runtime.silu_and_mul(input_tensor)

    assert result is None
    assert runtime.diagnostics()["counters"] == {"fallback.silu_and_mul.non_hpu_tensor": 1}


def test_strict_silu_and_mul_rejects_non_hpu_tensor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")
    input_tensor = torch.zeros(2, 16, dtype=torch.bfloat16)

    with pytest.raises(runtime.FastPathUnavailable, match="silu_and_mul: non_hpu_tensor"):
        runtime.silu_and_mul(input_tensor)


def test_silu_and_mul_rejects_odd_input_width():
    class FakeHpuTensor:
        device = torch.device("hpu")
        ndim = 2
        shape = (2, 15)
        dtype = torch.bfloat16

        @staticmethod
        def numel():
            return 30

        @staticmethod
        def is_contiguous():
            return True

    assert runtime._silu_and_mul_rejection_reason(FakeHpuTensor()) == "shape_mismatch"


def _cpu_gdn_inputs(batch: int = 1):
    return (
        torch.zeros(2, 48, 128, 128, dtype=torch.float32),
        torch.zeros(batch, 10240, dtype=torch.bfloat16),
        torch.zeros(batch, 48, dtype=torch.bfloat16),
        torch.zeros(batch, 48, dtype=torch.bfloat16),
        torch.zeros(48, dtype=torch.float32),
        torch.zeros(48, dtype=torch.float32),
        torch.arange(batch, dtype=torch.int32),
    )


def _cpu_gdn_conv_inputs(batch: int = 1):
    return (
        torch.zeros(2, 3, 10240, dtype=torch.bfloat16),
        *_cpu_gdn_inputs(batch),
        torch.zeros(4, 10240, dtype=torch.bfloat16),
    )


def test_off_mode_leaves_gdn_decode_on_vendor_path():
    result = runtime.gdn_decode_packed(*_cpu_gdn_inputs())

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_packed.off": 1,
    }


def test_hybrid_gdn_decode_applies_performance_gate_before_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_packed(*_cpu_gdn_inputs())

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_packed.performance_gate": 1,
    }


def test_strict_gdn_decode_rejects_non_hpu_tensor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")

    with pytest.raises(
        runtime.FastPathUnavailable,
        match="gdn_decode_packed: non_hpu_tensor",
    ):
        runtime.gdn_decode_packed(*_cpu_gdn_inputs())


def test_off_mode_leaves_fused_gdn_decode_on_vendor_path():
    result = runtime.gdn_decode_conv_packed(*_cpu_gdn_conv_inputs())

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_conv_packed.off": 1,
    }


def test_hybrid_fused_gdn_decode_stays_behind_performance_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_conv_packed(*_cpu_gdn_conv_inputs(batch=8))

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_conv_packed.performance_gate": 1,
    }


def test_strict_fused_gdn_decode_rejects_non_hpu_tensor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")

    with pytest.raises(
        runtime.FastPathUnavailable,
        match="gdn_decode_conv_packed: non_hpu_tensor",
    ):
        runtime.gdn_decode_conv_packed(*_cpu_gdn_conv_inputs())


def test_gdn_decode_validator_accepts_only_canonical_specialization():
    class FakeHpuTensor:
        device = torch.device("hpu")

        def __init__(self, shape, dtype, contiguous=True):
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = dtype
            self._contiguous = contiguous

        def is_contiguous(self):
            return self._contiguous

    tensors = (
        FakeHpuTensor((2, 48, 128, 128), torch.float32),
        FakeHpuTensor((1, 10240), torch.bfloat16),
        FakeHpuTensor((1, 48), torch.bfloat16),
        FakeHpuTensor((1, 48), torch.bfloat16),
        FakeHpuTensor((48,), torch.float32),
        FakeHpuTensor((48,), torch.float32),
        FakeHpuTensor((1,), torch.int32),
    )
    assert runtime._gdn_decode_packed_rejection_reason(*tensors) is None

    wrong_dtype = (*tensors[:-1], FakeHpuTensor((1,), torch.int64))
    assert runtime._gdn_decode_packed_rejection_reason(*wrong_dtype) == "unsupported_dtype"

    non_contiguous = (*tensors[:-1], FakeHpuTensor((1,), torch.int32, False))
    assert runtime._gdn_decode_packed_rejection_reason(*non_contiguous) == "non_contiguous"


@pytest.mark.skipif(shutil.which("tpc-clang") is None, reason="tpc-clang is not installed")
def test_gdn_decode_triton_ast_compiles_to_canonical_artifact():
    from vllm_gaudi.ops.triton_gaudi.kernels import _compile_gdn_decode_packed

    artifact = _compile_gdn_decode_packed(16)
    manifest = artifact.manifest

    assert artifact.elf.startswith(b"\x7fELF")
    assert manifest["kind"] == "gdn_decode_packed"
    assert manifest["input_args"] == [0, 1, 2, 3, 4, 5, 6]
    assert manifest["output_args"] == [7]
    assert manifest["index_space"] == {
        "rank": 3,
        "block_size": 16,
        "vector_lanes": 64,
        "program_id_axes": [0, 1, 2],
    }
    assert manifest["parameters"]["mutates_arg"] == 0
    assert manifest["parameters"]["state_slots_arg"] == 8


def test_hybrid_split_gdn_decode_keeps_small_batch_on_vendor_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_conv_split_packed(
        *_cpu_gdn_conv_inputs(batch=1))

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_conv_split_packed.performance_gate": 1,
    }


def test_hybrid_split_gdn_decode_falls_back_for_invalid_large_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_conv_split_packed(
        *_cpu_gdn_conv_inputs(batch=8))

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "fallback.gdn_decode_conv_split_packed.non_hpu_tensor": 1,
    }


def test_strict_split_gdn_decode_rejects_non_hpu_tensor(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")

    with pytest.raises(
        runtime.FastPathUnavailable,
        match="gdn_decode_conv_split_packed: non_hpu_tensor",
    ):
        runtime.gdn_decode_conv_split_packed(*_cpu_gdn_conv_inputs(batch=8))
