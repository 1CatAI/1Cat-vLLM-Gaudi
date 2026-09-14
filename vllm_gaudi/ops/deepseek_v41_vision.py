# SPDX-License-Identifier: Apache-2.0
"""Static image-span assembly for HPU; processor metadata stays on the host."""

import torch


def image_span_indices(roles, *, image, start, newline, end, image_rows):
    """Index [start, newline, end, *image] without device boolean indexing."""
    special = {start: 0, newline: 1, end: 2}
    result, count = [], 0
    for role in roles:
        if role == image:
            result.append(count + 3)
            count += 1
        elif role in special:
            result.append(special[role])
        else:
            raise ValueError("Unknown role in the frozen V4.1 image span")
    if count != image_rows:
        raise ValueError("Image encoder rows do not match the processor's placeholder span")
    return result


def assemble_image_span(features, special, indices):
    if special.shape != (3, features.shape[-1]):
        raise ValueError("Image special embeddings do not match the language hidden size")
    index = torch.tensor(indices, device="cpu", dtype=torch.int64).to(features.device)
    return torch.cat((special, features), dim=0).index_select(0, index)


def scatter_image_embeddings(output, is_embed):
    """Preserve upstream placeholder masks with a fixed-size index_copy."""
    if is_embed is None:
        return output
    mask = is_embed.detach().to(device="cpu", dtype=torch.bool)
    selected = mask.nonzero().flatten()
    if selected.numel() != output.shape[0]:
        raise ValueError("Image placeholder mask and encoded row count differ")
    if selected.numel() == mask.numel():
        return output
    scattered = output.new_full((mask.numel(), output.shape[-1]), float("nan"))
    scattered.index_copy_(0, selected.to(output.device), output)
    return scattered
