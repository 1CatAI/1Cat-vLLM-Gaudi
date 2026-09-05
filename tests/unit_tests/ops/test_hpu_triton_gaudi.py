# SPDX-License-Identifier: Apache-2.0

import operator
import shutil
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.triton_gaudi import runtime


def _fake_silu_and_mul(input_tensor, artifact_hash, block_size, n_cols, rows):
    return input_tensor


def _fake_dynamic_quant(input_tensor, artifact_hash, block_size, n_cols, rows):
    return input_tensor, input_tensor


def _fake_silu_and_mul_dynamic_quant(
    input_tensor,
    artifact_hash,
    block_size,
    n_cols,
    rows,
):
    return input_tensor, input_tensor


def _make_silu_dynamic_quant_graph(
    *,
    extra_silu_user: bool = False,
    quant_n_cols: int = 3584,
) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    input_tensor = graph.placeholder("input_tensor")
    silu = graph.call_function(
        _fake_silu_and_mul,
        args=(input_tensor, "a" * 64, 128, 3584, 8),
    )
    logical_view = graph.call_function(
        torch.ops.aten.view.default,
        args=(silu, [8, 3584]),
    )
    flattened = graph.call_function(
        torch.ops.aten.view.default,
        args=(logical_view, [-1]),
    )
    quantized_and_scale = graph.call_function(
        _fake_dynamic_quant,
        args=(flattened, "b" * 64, 4096, quant_n_cols, 8),
    )
    quantized = graph.call_function(operator.getitem, args=(quantized_and_scale, 0))
    scale = graph.call_function(operator.getitem, args=(quantized_and_scale, 1))
    outputs = (quantized, scale, silu) if extra_silu_user else (quantized, scale)
    graph.output(outputs)
    return torch.fx.GraphModule({}, graph)


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


def test_eager_hybrid_fused_add_rms_norm_keeps_candidate_on_vendor_path(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    class FakeHpuTensor:
        device = torch.device("hpu")
        ndim = 2
        shape = (8, 5120)
        dtype = torch.bfloat16

        @staticmethod
        def numel():
            return 8 * 5120

        @staticmethod
        def is_contiguous():
            return True

    class FakeHpuWeight(FakeHpuTensor):
        ndim = 1
        shape = (5120, )

        @staticmethod
        def numel():
            return 5120

    hidden = FakeHpuTensor()
    assert (runtime.fused_add_rms_norm(
        hidden,
        FakeHpuTensor(),
        FakeHpuWeight(),
        1.0e-6,
    ) is None)
    assert runtime.diagnostics()["counters"] == {
        "vendor.fused_add_rms_norm.performance_gate": 1,
    }


def test_strict_fused_add_rms_norm_rejects_non_hpu_tensor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")
    hidden = torch.zeros(2, 8, dtype=torch.bfloat16)

    with pytest.raises(runtime.FastPathUnavailable, match="fused_add_rms_norm: non_hpu_tensor"):
        runtime.fused_add_rms_norm(hidden, hidden, torch.ones(8, dtype=torch.bfloat16), 1.0e-6)


def test_off_mode_leaves_dynamic_quant_on_vendor_path():
    input_tensor = torch.zeros(2, 128, dtype=torch.bfloat16)

    result = runtime.dynamic_quant(input_tensor)

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.dynamic_quant.off": 1,
    }


def test_hybrid_dynamic_quant_records_non_hpu_fallback(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")
    input_tensor = torch.zeros(2, 128, dtype=torch.bfloat16)

    result = runtime.dynamic_quant(input_tensor)

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "fallback.dynamic_quant.non_hpu_tensor": 1,
    }


def test_strict_dynamic_quant_rejects_non_hpu_tensor(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")
    input_tensor = torch.zeros(2, 128, dtype=torch.bfloat16)

    with pytest.raises(
            runtime.FastPathUnavailable,
            match="dynamic_quant: non_hpu_tensor",
    ):
        runtime.dynamic_quant(input_tensor)


def test_dynamic_quant_validator_keeps_prefill_outside_fast_path():

    class FakeHpuTensor:
        device = torch.device("hpu")
        ndim = 2
        shape = (33, 4096)
        dtype = torch.bfloat16

        @staticmethod
        def numel():
            return 33 * 4096

        @staticmethod
        def is_contiguous():
            return True

    assert runtime._dynamic_quant_rejection_reason(FakeHpuTensor()) == "prefill_shape"


def test_eager_hybrid_dynamic_quant_keeps_candidate_on_vendor_path(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    class FakeHpuTensor:
        device = torch.device("hpu")
        ndim = 2
        shape = (8, 4096)
        dtype = torch.bfloat16

        @staticmethod
        def numel():
            return 8 * 4096

        @staticmethod
        def is_contiguous():
            return True

    assert runtime.dynamic_quant(FakeHpuTensor()) is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.dynamic_quant.performance_gate": 1,
    }


def test_off_mode_leaves_fused_silu_dynamic_quant_on_vendor_path():
    input_tensor = torch.zeros(2, 512, dtype=torch.bfloat16)

    result = runtime.silu_and_mul_dynamic_quant(input_tensor)

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.silu_and_mul_dynamic_quant.off": 1,
    }


def test_strict_fused_silu_dynamic_quant_rejects_non_hpu_tensor(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")
    input_tensor = torch.zeros(2, 512, dtype=torch.bfloat16)

    with pytest.raises(
            runtime.FastPathUnavailable,
            match="silu_and_mul_dynamic_quant: non_hpu_tensor",
    ):
        runtime.silu_and_mul_dynamic_quant(input_tensor)


def test_fused_silu_dynamic_quant_rejects_width_above_vlm_fast_path():

    class FakeHpuTensor:
        device = torch.device("hpu")
        ndim = 2
        shape = (8, 8194)
        dtype = torch.bfloat16

        @staticmethod
        def numel():
            return 8 * 8194

        @staticmethod
        def is_contiguous():
            return True

    assert runtime._silu_and_mul_dynamic_quant_rejection_reason(FakeHpuTensor()) == "unsupported_size"


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


def test_compile_hybrid_mode_keeps_ungated_kernels_on_vendor_path(monkeypatch: pytest.MonkeyPatch, ):
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
    dynamic_quant_result = runtime.dynamic_quant(hidden)
    silu_result = runtime.silu_and_mul(hidden)
    fused_silu_quant_result = runtime.silu_and_mul_dynamic_quant(hidden)
    gdn_result = runtime.gdn_decode_packed(*_cpu_gdn_inputs())

    assert rms_result is None
    assert dynamic_quant_result is None
    assert silu_result is None
    assert fused_silu_quant_result is None
    assert gdn_result is None
    assert runtime.diagnostics()["counters"] == {}


def test_compile_hybrid_gdn_keeps_candidate_on_vendor_path(monkeypatch: pytest.MonkeyPatch, ):
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


def test_eager_hybrid_gdn_keeps_candidate_on_vendor_path(monkeypatch: pytest.MonkeyPatch, ):
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


def test_eager_hybrid_silu_and_mul_keeps_candidate_on_vendor_path(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    class FakeHpuTensor:
        device = torch.device("hpu")
        ndim = 2
        shape = (8, 34816)
        dtype = torch.bfloat16

        @staticmethod
        def numel():
            return 8 * 34816

        @staticmethod
        def is_contiguous():
            return True

    assert runtime.silu_and_mul(FakeHpuTensor()) is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.silu_and_mul.performance_gate": 1,
    }


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


def test_hybrid_gdn_decode_applies_performance_gate_before_validation(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_packed(*_cpu_gdn_inputs())

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_packed.performance_gate": 1,
    }


def test_strict_gdn_decode_rejects_non_hpu_tensor(monkeypatch: pytest.MonkeyPatch, ):
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


def test_hybrid_fused_gdn_decode_stays_behind_performance_gate(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_conv_packed(*_cpu_gdn_conv_inputs(batch=8))

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_conv_packed.performance_gate": 1,
    }


def test_strict_fused_gdn_decode_rejects_non_hpu_tensor(monkeypatch: pytest.MonkeyPatch, ):
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
        FakeHpuTensor((48, ), torch.float32),
        FakeHpuTensor((48, ), torch.float32),
        FakeHpuTensor((1, ), torch.int32),
    )
    assert runtime._gdn_decode_packed_rejection_reason(*tensors) is None

    wrong_dtype = (*tensors[:-1], FakeHpuTensor((1, ), torch.int64))
    assert runtime._gdn_decode_packed_rejection_reason(*wrong_dtype) == "unsupported_dtype"

    non_contiguous = (*tensors[:-1], FakeHpuTensor((1, ), torch.int32, False))
    assert runtime._gdn_decode_packed_rejection_reason(*non_contiguous) == "non_contiguous"


def test_silu_dynamic_quant_pass_fuses_exclusive_view_chain(monkeypatch: pytest.MonkeyPatch, ):
    from vllm_gaudi.ops.triton_gaudi import fusion, kernels

    graph_module = _make_silu_dynamic_quant_graph()
    monkeypatch.setattr(
        fusion,
        "_resolve_triton_ops",
        lambda: (
            _fake_silu_and_mul,
            _fake_dynamic_quant,
            _fake_silu_and_mul_dynamic_quant,
        ),
    )
    monkeypatch.setattr(
        kernels,
        "_prepare_silu_and_mul_dynamic_quant",
        lambda n_cols: ("f" * 64, 4096),
    )

    assert fusion.pass_fuse_triton_gaudi_silu_dynamic_quant(SimpleNamespace(graph_module=graph_module))

    call_targets = [node.target for node in graph_module.graph.nodes if node.op == "call_function"]
    assert _fake_silu_and_mul not in call_targets
    assert _fake_dynamic_quant not in call_targets
    assert call_targets.count(_fake_silu_and_mul_dynamic_quant) == 1
    fused = next(node for node in graph_module.graph.nodes if node.target is _fake_silu_and_mul_dynamic_quant)
    assert fused.args[1:] == ("f" * 64, 4096, 3584, 8)
    assert fused.args[0].op == "placeholder"


@pytest.mark.parametrize(
    ("extra_silu_user", "quant_n_cols"),
    ((True, 3584), (False, 4096)),
)
def test_silu_dynamic_quant_pass_rejects_unsafe_graphs(
    monkeypatch: pytest.MonkeyPatch,
    extra_silu_user: bool,
    quant_n_cols: int,
):
    from vllm_gaudi.ops.triton_gaudi import fusion, kernels

    graph_module = _make_silu_dynamic_quant_graph(
        extra_silu_user=extra_silu_user,
        quant_n_cols=quant_n_cols,
    )
    monkeypatch.setattr(
        fusion,
        "_resolve_triton_ops",
        lambda: (
            _fake_silu_and_mul,
            _fake_dynamic_quant,
            _fake_silu_and_mul_dynamic_quant,
        ),
    )
    prepare_calls = []
    monkeypatch.setattr(
        kernels,
        "_prepare_silu_and_mul_dynamic_quant",
        lambda n_cols: prepare_calls.append(n_cols),
    )

    assert not fusion.pass_fuse_triton_gaudi_silu_dynamic_quant(SimpleNamespace(graph_module=graph_module))
    assert prepare_calls == []
    call_targets = [node.target for node in graph_module.graph.nodes if node.op == "call_function"]
    assert _fake_silu_and_mul in call_targets
    assert _fake_dynamic_quant in call_targets
    assert _fake_silu_and_mul_dynamic_quant not in call_targets


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


@pytest.mark.skipif(shutil.which("tpc-clang") is None, reason="tpc-clang is not installed")
def test_split_gdn_triton_ast_emits_stateful_two_kernel_artifacts():
    from vllm_gaudi.ops.triton_gaudi.kernels import (
        QK_CONV_TILE,
        _compile_gdn_decode_value_conv_packed,
        _compile_gdn_qk_conv_packed,
    )

    qk_manifest = _compile_gdn_qk_conv_packed().manifest
    assert qk_manifest["input_args"] == [0, 1, 2, 3]
    assert qk_manifest["output_args"] == [4]
    assert qk_manifest["index_space"]["block_size"] == QK_CONV_TILE == 256
    assert qk_manifest["index_space"]["vector_lanes"] == 128
    assert qk_manifest["parameters"]["mutates_args"] == [0]
    assert qk_manifest["parameters"]["conv_slots_arg"] == 5
    assert qk_manifest["access_patterns"][0]["role"] == "mutable_input"
    assert qk_manifest["access_patterns"][4]["role"] == "output"

    value_manifest = _compile_gdn_decode_value_conv_packed(16).manifest
    assert value_manifest["input_args"] == list(range(10))
    assert value_manifest["output_args"] == [10]
    assert value_manifest["parameters"]["mutates_args"] == [0, 1]
    assert value_manifest["parameters"]["conv_slots_arg"] == 11
    assert value_manifest["parameters"]["state_slots_arg"] == 12
    assert [pattern["role"] for pattern in value_manifest["access_patterns"][:2]] == [
        "mutable_input",
        "mutable_input",
    ]
    assert value_manifest["access_patterns"][10]["role"] == "output"


@pytest.mark.skipif(shutil.which("tpc-clang") is None, reason="tpc-clang is not installed")
def test_dynamic_quant_triton_ast_compiles_mlp_width_to_canonical_artifact():
    from vllm_gaudi.ops.triton_gaudi.kernels import _compile_dynamic_quant

    artifact, block_size = _compile_dynamic_quant(11008)
    manifest = artifact.manifest

    assert artifact.elf.startswith(b"\x7fELF")
    assert block_size == 16384
    assert manifest["kind"] == "dynamic_quant"
    assert manifest["input_args"] == [0]
    assert manifest["output_args"] == [1, 2]
    assert [argument["dtype"] for argument in manifest["arguments"]] == [
        "bf16",
        "fp8e4nv",
        "f32",
    ]
    assert manifest["index_space"] == {
        "rank": 1,
        "block_size": 16384,
        "vector_lanes": 256,
        "program_id_axes": [0],
    }
    assert manifest["parameters"] == {
        "n_cols": 11008,
        "fp8_max": 240.0,
        "scale_epsilon": 1.0e-8,
        "vlm_bytes": 0,
    }


@pytest.mark.skipif(shutil.which("tpc-clang") is None, reason="tpc-clang is not installed")
def test_fused_silu_dynamic_quant_triton_ast_compiles_to_canonical_artifact():
    from vllm_gaudi.ops.triton_gaudi.kernels import (
        _compile_silu_and_mul_dynamic_quant, )

    artifact, block_size = _compile_silu_and_mul_dynamic_quant(3584)
    manifest = artifact.manifest

    assert artifact.elf.startswith(b"\x7fELF")
    assert block_size == 4096
    assert manifest["kind"] == "silu_and_mul_dynamic_quant"
    assert manifest["input_args"] == [0]
    assert manifest["output_args"] == [1, 2]
    assert [argument["dtype"] for argument in manifest["arguments"]] == [
        "bf16",
        "fp8e4nv",
        "f32",
    ]
    assert manifest["index_space"] == {
        "rank": 1,
        "block_size": 4096,
        "vector_lanes": 256,
        "program_id_axes": [0],
    }
    assert manifest["parameters"] == {
        "n_cols": 3584,
        "input_row_stride": 7168,
        "fp8_max": 240.0,
        "scale_epsilon": 1.0e-8,
        "vlm_bytes": 7168,
    }


def test_hybrid_split_gdn_decode_keeps_small_batch_on_vendor_path(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_conv_split_packed(*_cpu_gdn_conv_inputs(batch=1))

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_conv_split_packed.performance_gate": 1,
    }


def test_hybrid_split_gdn_decode_keeps_large_batch_on_vendor_path(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_conv_split_packed(*_cpu_gdn_conv_inputs(batch=32))

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "vendor.gdn_decode_conv_split_packed.performance_gate": 1,
    }


def test_hybrid_split_gdn_decode_falls_back_for_invalid_large_batch(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "hybrid")

    result = runtime.gdn_decode_conv_split_packed(*_cpu_gdn_conv_inputs(batch=8))

    assert result is None
    assert runtime.diagnostics()["counters"] == {
        "fallback.gdn_decode_conv_split_packed.non_hpu_tensor": 1,
    }


def test_strict_split_gdn_decode_rejects_non_hpu_tensor(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")

    with pytest.raises(
            runtime.FastPathUnavailable,
            match="gdn_decode_conv_split_packed: non_hpu_tensor",
    ):
        runtime.gdn_decode_conv_split_packed(*_cpu_gdn_conv_inputs(batch=8))


def test_compile_strict_split_gdn_rejects_unsafe_graph_batch(monkeypatch: pytest.MonkeyPatch, ):
    monkeypatch.setenv("VLLM_HPU_TRITON_MODE", "strict")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(runtime, "_prepared", True)
    monkeypatch.setattr(runtime, "_available", True)

    with pytest.raises(
            runtime.FastPathUnavailable,
            match="unsupported_graph_batch",
    ):
        runtime.gdn_decode_conv_split_packed(*_cpu_gdn_conv_inputs(batch=12))


def test_bridge_pass_api_requires_gdn_reinplace():
    from vllm_gaudi.ops.triton_gaudi import fusion

    incomplete = SimpleNamespace(
        OptimizationPassPlacement=object(),
        register_pass_at_optimization_pass=lambda *args: None,
    )

    with pytest.raises(RuntimeError, match="pass_reinplace_triton_gaudi_gdn_decode"):
        fusion._validate_bridge_pass_api(incomplete)


def test_bridge_gdn_graph_policy_requires_shape_gate():
    from vllm_gaudi.ops.triton_gaudi import fusion

    with pytest.raises(RuntimeError, match="shape-gated GDN placement"):
        fusion._validate_bridge_gdn_graph_policy(SimpleNamespace())


def test_bridge_gdn_graph_policy_accepts_shape_gate():
    from vllm_gaudi.ops.triton_gaudi import fusion

    supported = SimpleNamespace(
        TRITON_GAUDI_GDN_BATCH_ARGS=dict(fusion._GDN_GRAPH_BATCH_ARGS),
        TRITON_GAUDI_GDN_BATCHES=fusion._GDN_GRAPH_BATCHES,
        TRITON_GAUDI_GRAPH_OPS={"dynamic_quant"},
        _is_supported_triton_gaudi_gdn_graph=lambda *args: True,
    )

    fusion._validate_bridge_gdn_graph_policy(supported)


def test_bridge_gdn_graph_policy_rejects_generic_placement():
    from vllm_gaudi.ops.triton_gaudi import fusion

    unsafe = SimpleNamespace(
        TRITON_GAUDI_GDN_BATCH_ARGS=dict(fusion._GDN_GRAPH_BATCH_ARGS),
        TRITON_GAUDI_GDN_BATCHES=fusion._GDN_GRAPH_BATCHES,
        TRITON_GAUDI_GRAPH_OPS={"gdn_decode_packed"},
        _is_supported_triton_gaudi_gdn_graph=lambda *args: True,
    )

    with pytest.raises(RuntimeError, match="unsafe generic GDN placement"):
        fusion._validate_bridge_gdn_graph_policy(unsafe)
