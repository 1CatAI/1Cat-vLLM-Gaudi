# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable decode convolution taps through the normal weight loader."""

import torch


def install_decode_conv_taps(owner: torch.nn.Module, convolution: torch.nn.Module) -> None:
    weight = convolution.weight
    if weight.ndim != 3 or weight.shape[1:] != (1, 4) or weight.dtype != torch.bfloat16:
        raise ValueError("Prepared GDN taps require BF16 [channels, 1, 4] weights")
    owner.register_buffer(
        "_hpu_decode_conv_taps",
        torch.empty((4, weight.shape[0]), dtype=torch.float32, device=weight.device),
        persistent=False,
    )
    owner._hpu_decode_conv_taps_ready = False
    original_loader = weight.weight_loader

    def refresh():
        parameter = convolution.weight
        taps = owner._hpu_decode_conv_taps
        if (parameter.shape != weight.shape or parameter.dtype != torch.bfloat16 or taps.dtype != torch.float32
                or taps.device != parameter.device):
            owner._hpu_decode_conv_taps_ready = False
            raise RuntimeError("Prepared GDN convolution weight allocation changed; rebuild the model")
        with torch.no_grad():
            taps.copy_(parameter[:, 0, :].transpose(0, 1).float())
        owner._hpu_decode_conv_taps_ready = True

    def load_and_prepare(parameter, loaded, *args, **kwargs):
        owner._hpu_decode_conv_taps_ready = False
        result = original_loader(parameter, loaded, *args, **kwargs)
        refresh()
        return result

    weight.weight_loader = load_and_prepare
    # Support the other normal parameter-loading entry point as well. Buffers
    # keep their bound addresses while their exact, widened values are refreshed.
    convolution.register_load_state_dict_post_hook(lambda _module, _incompatible: refresh())
