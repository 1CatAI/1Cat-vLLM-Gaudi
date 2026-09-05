# SPDX-License-Identifier: Apache-2.0
"""Experimental dense prefill MLP pipeline with complete TP reductions."""

from itertools import islice

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.models.qwen3_next import Qwen3NextModel


class HpuQwen3ChunkedMLP(Qwen2MoeMLP):
    """Overlap a chunk's reduction with the next chunk's entire MLP."""

    def forward(self, x):
        # Restrict the experimental path to single-sequence prefill. Decode,
        # batched requests and TP1 retain the original implementation.
        token_dim = x.ndim - 2
        should_chunk = (self._hpu_mlp_chunks > 1 and self.down_proj.tp_size == 2 and self.down_proj.reduce_results
                        and x.ndim in (2, 3) and (x.ndim == 2 or x.shape[0] == 1)
                        and x.shape[token_dim] >= self._hpu_mlp_chunk_threshold)
        if should_chunk:
            metadata = get_forward_context().attn_metadata
            seq_lens = getattr(metadata, "seq_lens_tensor", None)
            should_chunk = (metadata is not None and bool(getattr(metadata, "is_prompt", False))
                            and seq_lens is not None and seq_lens.shape[0] == 1)
        if not should_chunk:
            return super().forward(x)

        torch._dynamo.graph_break()
        outputs = []
        handles = []
        for input_chunk in torch.chunk(x, self._hpu_mlp_chunks, dim=token_dim):
            torch._dynamo.graph_break()
            gate_up, _ = self.gate_up_proj(input_chunk.contiguous())
            activated = self.act_fn(gate_up)
            projection = self.down_proj
            # Bypass only the synchronous wrapper, never the actual reduction.
            bias = None if projection.tp_rank > 0 or projection.skip_bias_add else projection.bias
            partial = projection.quant_method.apply(projection, activated, bias)
            torch._dynamo.graph_break()
            handles.append(torch.distributed.all_reduce(partial, group=get_tp_group().device_group, async_op=True))
            outputs.append(partial)

        for handle in handles:
            handle.wait()
        torch._dynamo.graph_break()
        return torch.cat(outputs, dim=token_dim)


def enable_hpu_qwen3_mlp_chunking(model, chunks: int, threshold: int) -> int:
    """Validate the entire dense topology before modifying any MLP instance."""
    if chunks < 1 or threshold < 2:
        raise ValueError("MLP chunks must be positive and the prefill threshold must be at least two")
    if chunks == 1:
        return 0
    if hasattr(model, "language_model"):
        model = model.language_model
    inner = getattr(model, "model", None)
    if not isinstance(inner, Qwen3NextModel):
        return 0
    mlps = []
    for layer in islice(inner.layers, inner.start_layer, inner.end_layer):
        mlp = getattr(layer, "mlp", None)
        if type(mlp) not in (Qwen2MoeMLP, HpuQwen3ChunkedMLP) or mlp.expert_gate is not None:
            return 0
        projection = mlp.down_proj
        if (projection.tp_size != 2 or not projection.reduce_results or not projection.input_is_parallel
                or projection.quant_method is None):
            return 0
        mlps.append(mlp)
    for mlp in mlps:
        mlp._hpu_mlp_chunks = chunks
        mlp._hpu_mlp_chunk_threshold = threshold
        mlp.__class__ = HpuQwen3ChunkedMLP
    return len(mlps)
