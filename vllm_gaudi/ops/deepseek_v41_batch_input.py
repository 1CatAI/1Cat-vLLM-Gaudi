# SPDX-License-Identifier: Apache-2.0
"""Fill an owned, pinned request metadata frame before its DMA submission."""


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
