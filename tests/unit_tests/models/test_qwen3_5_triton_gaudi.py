# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import vllm_gaudi.models.qwen3_5 as qwen3_5


def _inputs():
    state = torch.zeros(2, 3)
    packed = torch.arange(12, dtype=torch.bfloat16).view(3, 4).t()
    gate_a = torch.arange(8, dtype=torch.bfloat16).view(2, 4).t()
    gate_b = gate_a.clone().t().contiguous().t()
    a_log = torch.arange(8, dtype=torch.float32).view(2, 4).t()
    dt_bias = a_log.clone().t().contiguous().t()
    indices = torch.arange(4, dtype=torch.int32).view(2, 2).t().reshape(-1)
    return state, packed, gate_a, gate_b, a_log, dt_bias, indices


def _fused_inputs():
    state, packed, gate_a, gate_b, a_log, dt_bias, indices = _inputs()
    conv_state = torch.zeros(2, 3, 4)
    conv_weight_t = torch.zeros(4, 4)
    return (
        conv_state,
        state,
        packed,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        indices,
        conv_weight_t,
    )


def test_qwen_gdn_adapter_is_noop_when_candidate_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(qwen3_5, "_triton_gdn_decode_packed", None)

    assert qwen3_5._try_triton_gdn_decode_packed(*_inputs()) is None


def test_qwen_gdn_adapter_forwards_contiguous_inputs_and_mutable_state(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = None
    expected = torch.ones(4, 2, 3, dtype=torch.bfloat16)

    def fake_kernel(*args):
        nonlocal captured
        captured = args
        args[0].add_(1)
        return expected

    monkeypatch.setattr(qwen3_5, "_triton_gdn_decode_packed", fake_kernel)
    inputs = _inputs()

    actual = qwen3_5._try_triton_gdn_decode_packed(*inputs)

    assert actual is expected
    assert captured is not None
    assert captured[0] is inputs[0]
    assert torch.equal(inputs[0], torch.ones_like(inputs[0]))
    assert all(tensor.is_contiguous() for tensor in captured[1:])


def test_qwen_gdn_adapter_fails_closed_without_state_indices(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(qwen3_5, "_triton_gdn_decode_packed", lambda *args: None)
    monkeypatch.setattr(qwen3_5, "_triton_gaudi_mode", "strict")
    inputs = (*_inputs()[:-1], None)

    with pytest.raises(RuntimeError, match="requires state_indices"):
        qwen3_5._try_triton_gdn_decode_packed(*inputs)


def test_qwen_gdn_adapter_hybrid_falls_back_without_state_indices(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(qwen3_5, "_triton_gdn_decode_packed", lambda *args: None)
    monkeypatch.setattr(qwen3_5, "_triton_gaudi_mode", "hybrid")
    inputs = (*_inputs()[:-1], None)

    assert qwen3_5._try_triton_gdn_decode_packed(*inputs) is None


def test_qwen_fused_gdn_adapter_forwards_both_mutable_caches(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = None
    expected = torch.ones(4, 2, 3, dtype=torch.bfloat16)

    def fake_kernel(*args):
        nonlocal captured
        captured = args
        args[0].add_(1)
        args[1].add_(2)
        return expected

    monkeypatch.setattr(
        qwen3_5,
        "_triton_gdn_decode_conv_packed",
        fake_kernel,
    )
    inputs = _fused_inputs()

    actual = qwen3_5._try_triton_gdn_decode_conv_packed(*inputs)

    assert actual is expected
    assert captured is not None
    assert captured[0] is inputs[0]
    assert captured[1] is inputs[1]
    assert torch.equal(inputs[0], torch.ones_like(inputs[0]))
    assert torch.equal(inputs[1], torch.full_like(inputs[1], 2))
    assert all(tensor.is_contiguous() for tensor in captured[2:])


def test_qwen_fused_gdn_adapter_fails_closed_without_weight(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        qwen3_5,
        "_triton_gdn_decode_conv_packed",
        lambda *args: None,
    )
    monkeypatch.setattr(qwen3_5, "_triton_gaudi_mode", "strict")
    inputs = (*_fused_inputs()[:-1], None)

    with pytest.raises(RuntimeError, match="transposed conv weight"):
        qwen3_5._try_triton_gdn_decode_conv_packed(*inputs)


def test_qwen_gdn_weight_loaders_materialize_fast_path_views(
    monkeypatch: pytest.MonkeyPatch,
):
    def copy_loader(param, loaded_weight):
        with torch.no_grad():
            param.copy_(loaded_weight)

    class FakeModelConfig:
        hf_text_config = None

        @staticmethod
        def get_mamba_chunk_size():
            return 64

    def fake_base_init(self):
        torch.nn.Module.__init__(self)
        self.model_config = FakeModelConfig()
        self.key_dim = 4
        self.value_dim = 4
        self.tp_size = 1
        self.conv_kernel_size = 4
        self.conv1d = torch.nn.Conv1d(
            4,
            4,
            self.conv_kernel_size,
            groups=4,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.dt_bias = torch.nn.Parameter(
            torch.empty(4, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.conv1d.weight.weight_loader = copy_loader
        self.dt_bias.weight_loader = copy_loader

    monkeypatch.setattr(
        qwen3_5.QwenGatedDeltaNetAttention,
        "__init__",
        fake_base_init,
    )
    monkeypatch.setattr(qwen3_5, "_triton_gaudi_mode", "hybrid")
    attention = qwen3_5.HPUGatedDeltaNetAttention()
    loaded_conv = torch.arange(
        attention.conv1d.weight.numel(),
        dtype=torch.bfloat16,
    ).view_as(attention.conv1d.weight)
    loaded_dt_bias = torch.arange(4, dtype=torch.bfloat16) + 0.5

    attention.conv1d.weight.weight_loader(
        attention.conv1d.weight,
        loaded_conv,
    )
    attention.dt_bias.weight_loader(attention.dt_bias, loaded_dt_bias)

    expected_conv = loaded_conv.view(4, 4).transpose(0, 1).contiguous()
    assert attention._triton_conv_weight_ready
    assert attention._triton_dt_bias_ready
    assert torch.equal(attention._triton_conv_weight_t, expected_conv)
    assert attention._triton_dt_bias_f32.dtype == torch.float32
    assert torch.equal(attention._triton_dt_bias_f32, loaded_dt_bias.float())
    assert "_triton_conv_weight_t" not in attention.state_dict()
    assert "_triton_dt_bias_f32" not in attention.state_dict()
