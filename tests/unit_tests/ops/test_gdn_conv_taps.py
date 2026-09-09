# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.gdn_conv_taps import install_decode_conv_taps


@pytest.mark.parametrize("bias_enabled", [False, True])
def test_prepared_taps_exact_recurrence_and_slot_reuse(bias_enabled):
    generator = torch.Generator().manual_seed(471)
    channels = 128
    weight = torch.randn(channels, 4, generator=generator).bfloat16()
    taps = weight.T.float().contiguous()
    bias = torch.randn(channels, generator=generator).bfloat16() if bias_enabled else None
    ordinary = torch.full((5, 3, channels), 71, dtype=torch.bfloat16)
    prepared = ordinary.clone()
    for step in range(96):
        slot = (1, 3, 1)[step // 32]
        if step % 32 == 0:
            ordinary[slot].zero_()
            prepared[slot].zero_()
        x = torch.randn(1, channels, generator=generator).bfloat16()
        if step % 7 == 0:
            x.zero_()
        args = (x, ordinary[slot:slot + 1], weight, bias, "silu")
        qsl = torch.tensor([0, 1], dtype=torch.int32)
        expected = hpu_causal_conv1d_update(*args, direct_state_layout=True, query_start_loc=qsl)
        actual = hpu_causal_conv1d_update(x,
                                          prepared[slot:slot + 1],
                                          weight,
                                          bias,
                                          "silu",
                                          direct_state_layout=True,
                                          prepared_tap_weights=taps,
                                          query_start_loc=qsl)
        assert torch.equal(actual, expected)
        assert torch.equal(prepared, ordinary)


def test_loading_refreshes_fixed_taps_without_persisting_duplicate_weights():
    owner = torch.nn.Module()
    owner.conv = torch.nn.Conv1d(8, 8, 4, groups=8, bias=False, dtype=torch.bfloat16)

    def original(parameter, loaded):
        with torch.no_grad():
            parameter.copy_(loaded)
        return "loaded"

    owner.conv.weight.weight_loader = original
    install_decode_conv_taps(owner, owner.conv)
    assert not owner._hpu_decode_conv_taps_ready
    pointer = owner._hpu_decode_conv_taps.data_ptr()
    for value in (1, -2, 3):
        loaded = torch.full_like(owner.conv.weight, value)
        assert owner.conv.weight.weight_loader(owner.conv.weight, loaded) == "loaded"
        assert owner._hpu_decode_conv_taps_ready
        assert torch.equal(owner._hpu_decode_conv_taps, loaded[:, 0, :].T.float())
        assert owner._hpu_decode_conv_taps.data_ptr() == pointer
    state = owner.state_dict()
    assert set(state) == {"conv.weight"}
    state["conv.weight"] = torch.full_like(owner.conv.weight, 5)
    owner.load_state_dict(state)
    assert torch.equal(owner._hpu_decode_conv_taps, state["conv.weight"][:, 0, :].T.float())
    assert owner._hpu_decode_conv_taps.data_ptr() == pointer


@pytest.mark.parametrize("invalid", ["dtype", "shape", "stride", "indirect", "speculative"])
def test_invalid_prepared_taps_fail_before_state_mutation(invalid):
    state = torch.randn(1, 3, 16).bfloat16()
    before = state.clone()
    weight = torch.randn(16, 4).bfloat16()
    taps = weight.T.float().contiguous()
    options = {"direct_state_layout": True, "query_start_loc": torch.tensor([0, 1], dtype=torch.int32)}
    if invalid == "dtype":
        taps = taps.bfloat16()
    elif invalid == "shape":
        taps = taps[:3]
    elif invalid == "stride":
        taps = weight.float().T
    elif invalid == "indirect":
        options["direct_state_layout"] = False
    else:
        options["num_accepted_tokens"] = torch.ones(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="Prepared convolution taps"):
        hpu_causal_conv1d_update(torch.randn(1, 16).bfloat16(), state, weight, prepared_tap_weights=taps, **options)
    assert torch.equal(state, before)
