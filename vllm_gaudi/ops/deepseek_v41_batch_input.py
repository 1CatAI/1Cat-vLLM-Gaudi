# SPDX-License-Identifier: Apache-2.0
"""Fill an owned, pinned request metadata frame before its DMA submission."""

import torch


def fill_request_metadata(host, requests, owners, *, input_ids=None):
    # This aliases the existing pinned allocation. Its completion owner and
    # subsequent nonblocking DMA are unchanged; no per-row Torch operations
    # or temporary device tensors are needed to populate host metadata.
    values = host.numpy()
    values[0].fill(0)
    values[1:].fill(-1)
    if input_ids is None:
        input_ids = (request.tokens[request.num_computed_tokens] for request in requests)
    rows = [(token, request.num_computed_tokens, owner.index)
            for token, request, owner in zip(input_ids, requests, owners, strict=True)]
    if rows:
        values[:, :len(rows)] = tuple(zip(*rows))


class RequestInputFrame:
    """Reuse input views and page selection until request ownership changes.

    The caller must retire the frame's previous consumer before prepare, and
    publish each owner's current scheduler pages before beginning the batch.
    Only page selection is cached; token IDs and positions advance every step.
    """

    def __init__(self, bank, host, metadata, pages):
        self.bank, self.host, self.metadata, self.pages = bank, host, metadata, pages
        self.inputs = metadata.unbind(0)
        self.page_key = None

    def prepare(self, requests, owners, *, input_ids=None):
        owners = tuple(owners)
        key = []
        for owner in owners:
            version = self.bank.page_versions.get(owner.index)
            if version is None or version[0] != owner.generation:
                raise RuntimeError("Request input requires the current owner's published pages")
            key.append((owner.index, version))
        if len(owners) < self.pages.shape[0]:
            # The existing clamped gather selects bank row zero for padding.
            # Retain its values too, including changes by another active lane.
            key.append((0, self.bank.page_versions.get(0)))
        key = tuple(key)
        fill_request_metadata(self.host, requests, owners, input_ids=input_ids)
        self.metadata.copy_(self.host, non_blocking=True)
        if key != self.page_key:
            torch.index_select(self.bank.pages, 0, self.inputs[2].clamp_min(0).long(), out=self.pages)
            self.page_key = key
        return self.inputs
