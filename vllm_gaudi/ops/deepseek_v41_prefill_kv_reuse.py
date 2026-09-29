# SPDX-License-Identifier: Apache-2.0
"""Transaction-scoped reuse of immutable compressed-source KV decoding.

Adapted from the lifetime contract in SGLang PR 40421, commit
44eb378e94610e4ccc6b990bf09c105f8c8f9ee9. Gaudi retains compiled BF16
codec/assembly regions instead of the CUDA paged dequantization kernel.
"""
from functools import lru_cache
from types import FunctionType

import torch
from vllm_gaudi import envs as gaudi_envs


def _decode(packed, table, logical, ratio):
    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
    width = 128 // ratio
    shift = width.bit_length() - 1
    blocks = table.index_select(0, torch.bitwise_right_shift(logical, shift).long())
    physical = blocks * width + torch.bitwise_and(logical, width - 1)
    return unpack_fp4(packed.index_select(0, physical.long()))


def _decode_compact(packed, table, logical, ratio):
    """Gather compressed rows first, then decode with affine TPC reads.

    The large resident cache has runtime-selected pages. A packed row is only
    288 bytes; the compact gather keeps the native decoder's source access
    bounded and gives its TPC work points contiguous producer data.
    """
    width = 128 // ratio
    shift = width.bit_length() - 1
    blocks = table.index_select(0, torch.bitwise_right_shift(logical, shift).long())
    physical = blocks * width + torch.bitwise_and(logical, width - 1)
    compressed_rows = packed.index_select(0, physical.long())
    return torch.ops.custom_op.custom_deepseek_v41_prefill_main_decode_gaudi2(compressed_rows, table, logical, 0)


def _decode_paged(packed, table, logical, ratio):
    """Keep runtime page lookup inside the codec instead of a large gather graph."""
    return torch.ops.custom_op.custom_deepseek_v41_prefill_main_decode_gaudi2(packed, table, logical, ratio)


def _assemble(main, swa, indices, selected):
    offset = swa.shape[0]
    return (torch.cat((swa, main), 0), torch.cat((indices, torch.where(selected >= 0, selected + offset, -1)),
                                                 -1).int())


@lru_cache(maxsize=48)
def _compiled(function, signature):
    entry = FunctionType(function.__code__.replace(co_name=f"prefill_shared_kv_{function.__name__}_{signature}"),
                         function.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def _signature(*values):
    return tuple(
        (tuple(v.shape), v.stride(), v.storage_offset(), v.dtype, v.device) if isinstance(v, torch.Tensor) else v
        for v in values)


class PrefillMainWorkspace:
    """Keep at most one decoded source per ratio in a single forward.

    A new generation drops all prior validity. Producers always refresh even
    when the packed allocation address is unchanged. Reuse layers retain the
    same source identity, page mapping and logical prefix. Tensor ownership
    keeps submitted consumers alive when the next forward replaces scratch.
    """

    def __init__(self):
        self.generation = -1
        self.entries = {}
        self.decodes = 0
        self.hits = 0

    def begin(self, generation):
        if type(generation) is not int or generation <= self.generation:
            raise ValueError("Prefill KV scratch requires a new monotonic generation")
        self.entries.clear()
        self.generation = generation

    def get(self, source, packed, table, logical, ratio, *, producer=False):
        if self.generation < 0 or ratio not in (1, 2) or not 1 <= logical.numel() <= 65536:
            raise ValueError("Prefill KV reuse requires an active generation and bounded ratio-1/2 prefix")
        if (packed.device.type != "hpu" or packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] != 288
                or table.dtype != torch.int32 or logical.dtype != torch.int32 or not packed.is_contiguous()
                or not table.is_contiguous() or not logical.is_contiguous()):
            raise ValueError("Prefill KV reuse requires explicit packed pages and contiguous I32 metadata")
        key = (source, packed.data_ptr(), tuple(packed.shape), table.data_ptr(), logical.numel(), ratio)
        cached = self.entries.get(ratio)
        if producer or cached is None or cached[0] != key:
            if gaudi_envs.VLLM_HPU_DSV41_PREFILL_MAIN_DECODE:
                native = getattr(torch.ops.custom_op, "custom_deepseek_v41_prefill_main_decode_gaudi2", None)
                if native is None:
                    raise RuntimeError("Prefill main-KV decode requested without the Gaudi2 native codec")
                value = _compiled(_decode_paged, _signature(packed, table, logical, ratio))(packed, table, logical,
                                                                                            ratio)
            else:
                value = _compiled(_decode, _signature(packed, table, logical, ratio))(packed, table, logical, ratio)
            self.entries[ratio] = (key, value)
            self.decodes += 1
        else:
            self.hits += 1
        return self.entries[ratio][1]

    def assemble(self, main, swa, indices, selected):
        return _compiled(_assemble, _signature(main, swa, indices, selected))(main, swa, indices, selected)

    def stats(self):
        return dict(generation=self.generation,
                    decodes=self.decodes,
                    hits=self.hits,
                    resident_bytes=sum(value.numel() * value.element_size() for _, value in self.entries.values()))
