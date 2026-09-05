# SPDX-License-Identifier: Apache-2.0
"""Experimental prefill pipeline spanning attention output and dense MLP."""

from itertools import islice

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextDecoderLayer, Qwen3NextModel

from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention
from vllm_gaudi.models.qwen3_mlp import HpuQwen3ChunkedMLP
from vllm_gaudi.ops.hpu_layernorm import HPUGemmaRMSNorm, HPURMSNorm


def can_pipeline_prefill(x, threshold):
    if x.ndim not in (2, 3) or (x.ndim == 3 and x.shape[0] != 1):
        return False
    if x.shape[-2] < threshold:
        return False
    metadata = get_forward_context().attn_metadata
    lengths = getattr(metadata, "seq_lens_tensor", None)
    return (metadata is not None and bool(getattr(metadata, "is_prompt", False)) and lengths is not None
            and lengths.shape[0] == 1)


def partial_projection(projection, x):
    bias = None if projection.tp_rank > 0 or projection.skip_bias_add else projection.bias
    return projection.quant_method.apply(projection, x, bias)


def pipeline_attention_mlp(core, residual, projection, norm, mlp, chunks, next_layer=None, return_prefetched=False):
    """Return reduced MLP outputs and updated residuals, both token-major.

    Every attention chunk is reduced before its norm/MLP consumes it. The
    attention core and its KV/SSM state updates have already completed normally.
    """
    core = core.reshape(-1, core.shape[-1])
    residual_flat = residual.reshape(-1, residual.shape[-1])
    core_chunks = torch.chunk(core, chunks, dim=0)
    residual_chunks = torch.chunk(residual_flat, chunks, dim=0)
    attention_outputs = []
    attention_work = []
    mlp_outputs = []
    mlp_work = []
    updated_residuals = []
    torch._dynamo.graph_break()
    for core_chunk in core_chunks:
        torch._dynamo.graph_break()
        partial = partial_projection(projection, core_chunk.contiguous())
        torch._dynamo.graph_break()
        attention_work.append(torch.distributed.all_reduce(partial, group=get_tp_group().device_group, async_op=True))
        attention_outputs.append(partial)

    for partial, work, residual_chunk in zip(attention_outputs, attention_work, residual_chunks):
        work.wait()
        torch._dynamo.graph_break()
        # The HPU fused norm expects a three-dimensional residual even when
        # the surrounding decoder uses flattened token-major activations.
        normalized, updated = norm(partial, residual_chunk.unsqueeze(0))
        gate_up, _ = mlp.gate_up_proj(normalized)
        activated = mlp.act_fn(gate_up)
        output = partial_projection(mlp.down_proj, activated)
        torch._dynamo.graph_break()
        mlp_work.append(torch.distributed.all_reduce(output, group=get_tp_group().device_group, async_op=True))
        mlp_outputs.append(output)
        updated_residuals.append(updated.reshape(-1, updated.shape[-1]))

    if next_layer is None:
        for work in mlp_work:
            work.wait()
        torch._dynamo.graph_break()
        result = torch.cat(mlp_outputs, dim=0), torch.cat(updated_residuals, dim=0)
        return (*result, None) if return_prefetched else result

    # Use completed MLP chunks immediately, while later reductions remain in
    # flight. Only token-local norm and input projections move across layers;
    # the next attention core still receives the full ordered sequence.
    next_hidden, next_residuals, first_projection, second_projection = [], [], [], []
    for output, work, updated in zip(mlp_outputs, mlp_work, updated_residuals):
        work.wait()
        torch._dynamo.graph_break()
        normalized, next_residual = next_layer.input_layernorm(output, updated.unsqueeze(0))
        if next_layer.layer_type == "linear_attention":
            projected, _ = next_layer.linear_attn.in_proj_qkvz(normalized)
            ba, _ = next_layer.linear_attn.in_proj_ba(normalized)
            second_projection.append(ba)
        else:
            projected, _ = next_layer.self_attn.qkv_proj(normalized)
        first_projection.append(projected)
        next_hidden.append(normalized)
        next_residuals.append(next_residual.reshape(-1, next_residual.shape[-1]))
    torch._dynamo.graph_break()
    projected = (torch.cat(first_projection, dim=0), )
    if second_projection:
        projected = (*projected, torch.cat(second_projection, dim=0))
    return torch.cat(next_hidden, dim=0), torch.cat(next_residuals, dim=0), projected


class HpuQwen3BoundaryDecoderLayer(Qwen3_5DecoderLayer):

    def forward(self, hidden_states, residual, positions=None, prefetched=None, return_prefetched=False, **kwargs):
        if self._hpu_boundary_chunks <= 1 or not can_pipeline_prefill(hidden_states, self._hpu_boundary_threshold):
            if prefetched is not None:
                raise RuntimeError("Prefetched projections require the active boundary pipeline")
            result = super().forward(hidden_states, residual, positions, **kwargs)
            return (*result, None) if return_prefetched else result
        hidden_shape = hidden_states.shape
        if prefetched is not None:
            # The previous layer already performed this input norm and both
            # input projections on completed chunks.
            pass
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        residual_shape = residual.shape
        if self.layer_type == "linear_attention":
            core = self.linear_attn(hidden_states=hidden_states, return_core=True, projected_input=prefetched)
            projection = self.linear_attn.out_proj
        else:
            core = self.self_attn(hidden_states=hidden_states,
                                  positions=positions,
                                  return_core=True,
                                  projected_input=prefetched)
            projection = self.self_attn.o_proj
        result = pipeline_attention_mlp(
            core,
            residual,
            projection,
            self.post_attention_layernorm,
            self.mlp,
            self._hpu_boundary_chunks,
            next_layer=getattr(self, "_hpu_prefetch_next", None) if return_prefetched else None,
            return_prefetched=return_prefetched)
        output, residual = result[:2]
        if return_prefetched:
            return output.reshape(hidden_shape), residual.reshape(residual_shape), result[2]
        return output.reshape(hidden_shape), residual.reshape(residual_shape)


def enable_hpu_qwen3_boundary_pipeline(model, chunks, threshold, prefetch=False):
    if chunks < 1 or threshold < 2:
        raise ValueError("Boundary chunks must be positive and prefill threshold at least two")
    if chunks == 1:
        return 0
    if hasattr(model, "language_model"):
        model = model.language_model
    inner = getattr(model, "model", None)
    if not isinstance(inner, Qwen3NextModel):
        return 0
    if prefetch and (inner.start_layer != 0 or inner.end_layer != len(inner.layers)
                     or getattr(inner, "aux_hidden_state_layers", ())):
        return 0
    layers = tuple(islice(inner.layers, inner.start_layer, inner.end_layer))
    for layer in layers:
        if type(layer) not in (Qwen3NextDecoderLayer, Qwen3_5DecoderLayer, HpuQwen3BoundaryDecoderLayer):
            return 0
        if layer.layer_scale or layer.use_attn_reduce_scatter_for_moe:
            return 0
        if type(layer.mlp) not in (Qwen2MoeMLP, HpuQwen3ChunkedMLP) or layer.mlp.expert_gate is not None:
            return 0
        if not all(
                isinstance(norm, (HPUGemmaRMSNorm, HPURMSNorm))
                for norm in (layer.input_layernorm, layer.post_attention_layernorm)):
            return 0
        if layer.layer_type == "linear_attention" and isinstance(layer.linear_attn, HPUGatedDeltaNetAttention):
            if prefetch and hasattr(layer.linear_attn, "in_proj_qkv"):
                return 0
            projection = layer.linear_attn.out_proj
        elif layer.layer_type == "full_attention" and isinstance(layer.self_attn, Qwen3NextAttention):
            projection = layer.self_attn.o_proj
        else:
            return 0
        for candidate in (projection, layer.mlp.down_proj):
            if (candidate.tp_size != 2 or not candidate.reduce_results or not candidate.input_is_parallel
                    or candidate.quant_method is None):
                return 0
    for layer in layers:
        layer._hpu_boundary_chunks = chunks
        layer._hpu_boundary_threshold = threshold
        layer.__class__ = HpuQwen3BoundaryDecoderLayer
    if prefetch:
        for index, layer in enumerate(layers):
            # Preserve sole parameter ownership by the original ModuleList.
            following = layers[index + 1] if index + 1 < len(layers) else None
            object.__setattr__(layer, "_hpu_prefetch_next", following)
        inner._hpu_boundary_prefetch = True
    return len(layers)
