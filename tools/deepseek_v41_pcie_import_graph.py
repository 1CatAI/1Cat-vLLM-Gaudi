# SPDX-License-Identifier: Apache-2.0
"""Reject cyclic device-context ownership before an experimental import."""


def importer_first_order(size, imports):
    edges = set(imports)
    if size < 2 or any(a == b or not 0 <= a < size or not 0 <= b < size for a, b in edges):
        raise ValueError('Invalid device import graph')
    waiting = [0] * size
    for _, exporter in edges:
        waiting[exporter] += 1
    ready = [rank for rank, count in enumerate(waiting) if not count]
    result = []
    while ready:
        rank = ready.pop(0)
        result.append(rank)
        for importer, exporter in edges:
            if importer == rank:
                waiting[exporter] -= 1
                if waiting[exporter] == 0:
                    ready.append(exporter)
    if len(result) != size:
        raise ValueError('Cyclic DMA-BUF imports can retain driver contexts after process death')
    return result


def star_imports(size):
    if not 2 <= size <= 4:
        raise ValueError('PCIe star probe accepts two through four ranks')
    imports = [(rank, 0) for rank in range(1, size)]
    order = importer_first_order(size, imports)
    if order[-1] != 0:
        raise ValueError('The exporting hub must retire last')
    return imports, order


def epoch_reached(observed, expected):
    """Wrap-safe comparison when peers are less than 2**31 epochs apart."""
    return ((observed - expected) & 0xffffffff) < 0x80000000
