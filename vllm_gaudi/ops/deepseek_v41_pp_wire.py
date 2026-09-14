# SPDX-License-Identifier: Apache-2.0
"""Pure tensor PP wire transforms shared by compiled and compatibility paths."""
import torch

HIDDEN_SLOTS = 4
HIDDEN_WIDTH = 5120
HIDDEN_ELEMENTS = HIDDEN_SLOTS * HIDDEN_WIDTH
WIRE_ELEMENTS = HIDDEN_ELEMENTS + HIDDEN_SLOTS * 4


def encode_pp_wire(hidden, pre_mix):
    if (hidden.dtype != torch.bfloat16 or hidden.ndim != 3
            or hidden.shape[1:] != (HIDDEN_SLOTS, HIDDEN_WIDTH)
            or pre_mix.dtype != torch.float32 or pre_mix.shape != hidden.shape[:2]):
        raise ValueError("Invalid V4.1 PP stage output contract")
    bits = pre_mix.contiguous().view(torch.int32).to(torch.int64)
    lanes = torch.stack(tuple((bits >> shift) & 255 for shift in (0, 8, 16, 24)), -1)
    return torch.cat((hidden.reshape(hidden.shape[0], HIDDEN_ELEMENTS),
                      lanes.to(torch.bfloat16).reshape(hidden.shape[0], HIDDEN_SLOTS * 4)), dim=1)


def decode_pre_mix(wire):
    encoded = wire.to(torch.int32).reshape(-1, 4).to(torch.int64)
    multipliers = torch.tensor([1, 1 << 8, 1 << 16, 1 << 24], dtype=torch.int64, device=wire.device)
    unsigned = (encoded * multipliers).sum(-1)
    signed = torch.where(unsigned >= (1 << 31), unsigned - (1 << 32), unsigned)
    return signed.to(torch.int32).view(torch.float32).reshape(wire.shape[0], HIDDEN_SLOTS)


def decode_pp_wire(wire):
    if wire.dtype != torch.bfloat16 or wire.ndim != 2 or wire.shape[1] != WIRE_ELEMENTS:
        raise ValueError("Invalid V4.1 PP wire contract")
    hidden = wire[:, :HIDDEN_ELEMENTS].reshape(wire.shape[0], HIDDEN_SLOTS, HIDDEN_WIDTH)
    pre = decode_pre_mix(wire[:, HIDDEN_ELEMENTS:].reshape(wire.shape[0], HIDDEN_SLOTS, 4))
    return hidden, pre
