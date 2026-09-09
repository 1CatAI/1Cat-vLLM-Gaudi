# SPDX-License-Identifier: Apache-2.0
"""Prepare the BF16 Gemma norm offset while loading immutable weights."""

import torch


def install_decode_gemma_weight(owner: torch.nn.Module, default_loader, retire) -> None:
    weight = owner.weight
    if weight.ndim != 1 or weight.dtype != torch.bfloat16:
        raise ValueError("Prepared Gemma norm requires a BF16 weight vector")
    owner.register_buffer("_hpu_decode_gemma_weight", torch.empty_like(weight), persistent=False)
    owner._hpu_decode_gemma_weight_ready = False
    shape = tuple(weight.shape)
    original_loader = getattr(weight, "weight_loader", default_loader)

    def validate():
        parameter = owner.weight
        target = owner._hpu_decode_gemma_weight
        if (parameter is not weight or tuple(parameter.shape) != shape or parameter.dtype != torch.bfloat16
                or target.dtype != torch.bfloat16 or tuple(target.shape) != shape or target.device != parameter.device):
            owner._hpu_decode_gemma_weight_ready = False
            raise RuntimeError("Prepared Gemma norm allocation changed; rebuild the model before loading")

    def before_load():
        validate()
        # The fixed weight is read by native programs. Retire them before any
        # loader mutation, including reloads through state_dict.
        owner._hpu_decode_gemma_weight_ready = False
        retire()

    def refresh():
        validate()
        with torch.no_grad():
            owner._hpu_decode_gemma_weight.copy_(owner.weight + 1.0)
        owner._hpu_decode_gemma_weight_ready = True

    def load_and_prepare(parameter, loaded, *args, **kwargs):
        if parameter is not owner.weight:
            raise RuntimeError("Prepared Gemma norm loader received a different parameter")
        before_load()
        result = original_loader(parameter, loaded, *args, **kwargs)
        refresh()
        return result

    weight.weight_loader = load_and_prepare
    owner.register_load_state_dict_pre_hook(lambda *_args: before_load())
    owner.register_load_state_dict_post_hook(lambda *_args: refresh())
