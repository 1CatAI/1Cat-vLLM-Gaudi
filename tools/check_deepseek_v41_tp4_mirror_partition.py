# SPDX-License-Identifier: Apache-2.0
"""Check query gathers -> sharded mirror scores -> candidate gather -> matrix consumer."""
from types import FunctionType, MethodType, SimpleNamespace

import torch

from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


@torch.inference_mode()
def check_mirror_partition(gather):
    generator = torch.Generator().manual_seed(939)
    keys = torch.randn(32768, 128, generator=generator).to(device="hpu", dtype=torch.bfloat16)
    query = torch.randn(1, 32, 128, generator=generator).to(device="hpu", dtype=torch.bfloat16)
    weights = torch.randn(1, 32, generator=generator).to(device="hpu", dtype=torch.bfloat16)
    rank = torch.distributed.get_rank()

    class Selection(torch.nn.Module):
        _merge_topk = staticmethod(PagedCSA2Attention._merge_topk)

        def __init__(self, ratio, columns, visible, blocks, partition):
            super().__init__()
            self.ratio, self.visible, self.blocks, self.partition = ratio, visible, blocks, partition
            self.tensor_parallel_size, self.index_heads, self.prefill_tp_rank = 4, 8, rank
            self.index_mirror_scores = True
            self.register_buffer("keys", keys[:32768 // ratio])
            self.cache = SimpleNamespace(index_mirror=self.keys)
            self.register_buffer("rows", torch.arange(columns, device="hpu", dtype=torch.int32))
            self.gather = gather

        def _uses_index_mirror(self, q):
            return True

        def _scores(self, positions, rows, q, weight):
            from vllm_gaudi.ops.deepseek_v41_index_mirror import mirror_index_tile
            return mirror_index_tile(q, weight, self.keys, positions, rows, self.ratio, self.index_heads)

        def forward(self, local_q, local_weight, position):
            q = self.gather(local_q, 1)
            weight = self.gather(local_weight, 1)
            rows, scores, blocks = PagedCSA2Attention._stream_topk(
                self, position, self.rows, q, weight, collect_blocks=self.blocks,
                visible_rows=self.visible, partition_mirror=self.partition)
            selected = self.keys.index_select(0, rows.flatten().clamp_min(0).long())
            consumer = torch.matmul(q, selected.T.contiguous()).sum(-1)
            return rows, scores, blocks, consumer

    cases = []
    for case, (ratio, columns, visible, blocks, ties) in enumerate([
            (1, 32768, 20480, True, False), (2, 16384, 10240, False, False),
            (1, 32768, 8192, True, True), (1, 32768, 24576, True, True),
            (2, 16384, 8192, False, True)]):
        functions = []
        for partition in (False, True):
            module = Selection(ratio, columns, visible, blocks, partition)
            method = module.forward
            code = method.__func__
            name = f"mirror40_{case}_{int(partition)}"
            entry = FunctionType(code.__code__.replace(co_name=name), code.__globals__, name,
                                 code.__defaults__, code.__closure__)
            functions.append(torch.compile(MethodType(entry, module), backend=make_backend(),
                                            fullgraph=True, dynamic=False))
        current_q = torch.zeros_like(query) if ties else query
        local_q = current_q[:, rank * 8:(rank + 1) * 8].clone().contiguous()
        local_weights = weights[:, rank * 8:(rank + 1) * 8].clone().contiguous()
        position = torch.tensor([min(16384, visible * ratio - 1)], dtype=torch.int32, device="hpu")
        wanted = functions[0](local_q, local_weights, position)
        observed = functions[1](local_q, local_weights, position)
        for expected, actual in zip(wanted, observed, strict=True):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        cases.append(dict(ratio=ratio, columns=columns, visible=visible, ties=ties, exact=True,
                          query_collectives_and_matrix_consumer=True))
    return dict(status="exact", cases=cases, timing_collected=False)
