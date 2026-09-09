# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from types import SimpleNamespace

from vllm_gaudi.ops.prepared_gemma_weight import install_decode_gemma_weight


def test_loader_retires_before_mutation_and_refreshes_fixed_buffer():
    owner = torch.nn.Module()
    owner.weight = torch.nn.Parameter(torch.zeros(5120, dtype=torch.bfloat16))
    retired = []

    def retire():
        retired.append(owner.weight.detach().clone())

    def loader(parameter, value):
        assert torch.equal(retired[-1], parameter)
        with torch.no_grad():
            parameter.copy_(value)
        return "loaded"

    install_decode_gemma_weight(owner, loader, retire)
    pointer = owner._hpu_decode_gemma_weight.data_ptr()
    assert not owner._hpu_decode_gemma_weight_ready
    generator = torch.Generator().manual_seed(473)
    for _ in range(4):
        values = torch.randn(5120, generator=generator).bfloat16()
        assert owner.weight.weight_loader(owner.weight, values) == "loaded"
        assert owner._hpu_decode_gemma_weight_ready
        assert torch.equal(owner._hpu_decode_gemma_weight, values + 1.0)
        assert owner._hpu_decode_gemma_weight.data_ptr() == pointer
    assert set(owner.state_dict()) == {"weight"}
    values = torch.full_like(owner.weight, -0.75)
    owner.load_state_dict({"weight": values})
    assert len(retired) == 5
    assert torch.equal(owner._hpu_decode_gemma_weight, values + 1.0)
    assert owner._hpu_decode_gemma_weight.data_ptr() == pointer


@pytest.mark.parametrize("failure", ["retire", "loader", "allocation"])
def test_failed_load_cannot_leave_ready_stale_weights(failure):
    owner = torch.nn.Module()
    owner.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))

    def retire():
        if failure == "retire":
            raise RuntimeError("in-flight failure")

    def loader(parameter, values):
        raise RuntimeError("loader failure")

    install_decode_gemma_weight(owner, loader, retire)
    owner._hpu_decode_gemma_weight_ready = True
    if failure == "allocation":
        owner.weight = torch.nn.Parameter(owner.weight.float())
        with pytest.raises(RuntimeError, match="allocation changed"):
            owner.load_state_dict({"weight": torch.ones_like(owner.weight)})
    else:
        with pytest.raises(RuntimeError, match="failure"):
            owner.weight.weight_loader(owner.weight, torch.ones_like(owner.weight))
    assert not owner._hpu_decode_gemma_weight_ready
    assert torch.equal(owner.weight, torch.zeros_like(owner.weight))


@pytest.mark.parametrize("is_prompt", [True, False])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("direct", [True, False])
def test_prepared_weight_only_enters_supported_decode(monkeypatch, is_prompt, rows, direct):
    import vllm.forward_context as forward_context
    import vllm_gaudi.extension.kernels as kernels
    from vllm_gaudi.ops.hpu_layernorm import HPUGemmaRMSNorm

    weight = torch.linspace(-0.75, 0.5, 5120).bfloat16()
    prepared = weight + 1.0
    norm = SimpleNamespace(weight=weight,
                           variance_epsilon=1e-6,
                           _hpu_prepared_gemma_weight=True,
                           _hpu_decode_gemma_weight_ready=True,
                           _hpu_decode_gemma_weight=prepared)
    calls = []

    class RMSNorm:

        @staticmethod
        def apply(x, w, epsilon):
            calls.append(w)
            return x

    monkeypatch.setattr(kernels, "rms_norm", lambda: RMSNorm)
    metadata = SimpleNamespace(is_prompt=is_prompt, direct_gdn_state=direct)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    HPUGemmaRMSNorm.forward_oot(norm, torch.ones(rows, 5120, dtype=torch.bfloat16))
    assert torch.equal(calls[0], prepared)
    assert (calls[0] is prepared) == (not is_prompt and rows == 1 and direct)
