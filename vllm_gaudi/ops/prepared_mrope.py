# SPDX-License-Identifier: Apache-2.0
"""Static MRoPE lane selection with independently changing position axes."""

import torch


def make_mrope_selection(rotary_dim, sections, device):
    if rotary_dim != 64 or tuple(sections) != (11, 11, 10):
        raise RuntimeError("Prepared MRoPE requires the qualified rotary dimensions and sections")
    channels = torch.arange(rotary_dim, dtype=torch.int64, device=device)
    frequency = channels % (rotary_dim // 2)
    axes = torch.where((frequency % 3 == 1) & (frequency < sections[1] * 3), 1, 0)
    axes = torch.where((frequency % 3 == 2) & (frequency < sections[2] * 3), 2, axes)
    return axes * rotary_dim + channels


def prepare_mrope_coefficients(cache, positions, selection):
    values = cache[positions].reshape(-1).index_select(0, selection).reshape(1, 64)
    cosine, sine = values.chunk(2, -1)
    return torch.cat((cosine, cosine), -1).unsqueeze(1), torch.cat((sine, sine), -1).unsqueeze(1)
