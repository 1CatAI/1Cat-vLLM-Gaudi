from itertools import islice

import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextModel as UpstreamQwen3NextModel,
    Qwen3NextSparseMoeBlock,
)
from vllm.sequence import IntermediateTensors
from vllm_gaudi.models.utils import sequence_parallel_chunk

HPU_QWEN3_LAYER_GROUP_MAX_BATCH_SIZE = 16


def _qwen3_inner_model(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model
    return getattr(model, "model", None)


def enable_hpu_qwen3_tp2_fused_ar_norm(model) -> int:
    """Move dense Qwen3 row-parallel reductions to following RMSNorms.

    Returns the number of all-reduce/RMSNorm boundaries enabled. No module is
    mutated when the model does not match the supported dense topology.
    """
    inner_model = _qwen3_inner_model(model)
    if not isinstance(inner_model, UpstreamQwen3NextModel):
        return 0
    layers = tuple(islice(inner_model.layers, inner_model.start_layer, inner_model.end_layer))
    if not layers or inner_model.start_layer != 0:
        return 0

    row_parallel_layers = []
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        output_projection = getattr(attention, "o_proj", None)
        if output_projection is None:
            attention = getattr(layer, "linear_attn", None)
            output_projection = getattr(attention, "out_proj", None)
        down_projection = getattr(getattr(layer, "mlp", None), "down_proj", None)
        if output_projection is None or down_projection is None:
            return 0
        if getattr(layer, "use_attn_reduce_scatter_for_moe", False):
            return 0
        row_parallel_layers.extend((output_projection, down_projection))

    from vllm_gaudi.ops.hpu_layernorm import HPUGemmaRMSNorm, HPURMSNorm

    norms = [getattr(inner_model, "norm", None)]
    for layer in layers:
        norms.extend((getattr(layer, "input_layernorm", None), getattr(layer, "post_attention_layernorm", None)))
    if not all(isinstance(norm, (HPURMSNorm, HPUGemmaRMSNorm)) for norm in norms):
        # Never disable a reduction unless its consumer implements the
        # deferred collective, including upstream normalization variants.
        return 0

    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime

    initialize_tp2_fused_ar_norm_runtime()
    inner_model._hpu_tp2_defer_embedding_reduce = True
    inner_model.embed_tokens._hpu_defer_tp2_reduce = True
    for projection in row_parallel_layers:
        projection.reduce_results = False
    for norm in norms:
        norm._hpu_tp2_fused_ar_norm = True
    return len(row_parallel_layers) + 1


class HpuQwen3DecoderLayerGroup(torch.nn.Module):
    """Run several decoder layers inside one torch.compile region."""

    def __init__(
        self,
        layers: tuple[torch.nn.Module, ...],
        final_norm: torch.nn.Module | None = None,
    ):
        super().__init__()
        # The model's ModuleList remains the sole owner of these layers so
        # state-dict and KV-cache layer names stay unchanged.
        object.__setattr__(self, "_layers", layers)
        object.__setattr__(self, "_final_norm", final_norm)

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self._layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        if self._final_norm is not None:
            hidden_states, residual = self._final_norm(hidden_states, residual)
        return hidden_states, residual


def build_hpu_qwen3_layer_groups(
    model: "HpuQwen3NextModel",
    group_size: int,
    final_norm: torch.nn.Module | None = None,
) -> tuple[HpuQwen3DecoderLayerGroup, ...]:
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")

    layers = tuple(islice(model.layers, model.start_layer, model.end_layer))
    groups = tuple(
        HpuQwen3DecoderLayerGroup(layers[start:start + group_size]) for start in range(0, len(layers), group_size))
    if groups and final_norm is not None:
        object.__setattr__(groups[-1], "_final_norm", final_norm)
    return groups


def compile_hpu_qwen3_layer_groups(
    model: "HpuQwen3NextModel",
    group_size: int,
    compile_fn,
    final_norm: torch.nn.Module | None = None,
) -> tuple[torch.nn.Module, ...]:
    groups = build_hpu_qwen3_layer_groups(model, group_size, final_norm)
    compiled_groups = tuple(compile_fn(group) for group in groups)
    object.__setattr__(model, "_hpu_compiled_layer_groups", compiled_groups)
    object.__setattr__(model, "_hpu_compiled_groups_include_final_norm", final_norm is not None)
    return compiled_groups


def can_compile_hpu_qwen3_layer_groups(
    group_size: int,
    tensor_parallel_size: int,
    aux_hidden_state_layers: tuple[int, ...] | list[int] | None,
) -> bool:
    return (group_size > 1 and tensor_parallel_size == 1 and not aux_hidden_state_layers)


def can_use_hpu_qwen3_layer_groups(
    layer_groups: tuple[torch.nn.Module, ...] | None,
    aux_hidden_state_layers: tuple[int, ...] | list[int] | None,
    attn_metadata,
    batch_size: int,
) -> bool:
    return (layer_groups is not None and not aux_hidden_state_layers and attn_metadata is not None
            and not bool(getattr(attn_metadata, "is_prompt", False))
            and (batch_size <= HPU_QWEN3_LAYER_GROUP_MAX_BATCH_SIZE
                 or bool(getattr(attn_metadata, "direct_gdn_state", False))))


def supports_hpu_qwen3_layer_group_compilation(
    tensor_parallel_size: int,
    tp2_fused_ar_norm: bool,
) -> bool:
    """Return whether grouped Qwen3 decode graphs can contain collectives."""
    return tensor_parallel_size == 1 or (tensor_parallel_size == 2 and tp2_fused_ar_norm)


class HpuQwen3NextModel(UpstreamQwen3NextModel):
    """Qwen3NextModel with residual initialized as zeros instead of None.

    The upstream Qwen3NextModel.forward() sets ``residual = None`` for the first
    rank, which creates a torch._dynamo type guard (None vs Tensor) that
    causes recompilation between layer 0 and layers 1+. Initializing
    residual as ``torch.zeros_like(hidden_states)`` eliminates this guard.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if getattr(self, "_hpu_tp2_defer_embedding_reduce", False) and inputs_embeds is not None:
                raise RuntimeError("TP2 fused embedding reduction does not support inputs_embeds")
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = torch.zeros_like(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        layer_groups = getattr(self, "_hpu_compiled_layer_groups", None)
        attn_metadata = get_forward_context().attn_metadata
        use_boundary_prefetch = False
        if getattr(self, "_hpu_boundary_prefetch", False):
            from vllm_gaudi.models.qwen3_boundary import can_pipeline_prefill
            first_layer = self.layers[self.start_layer]
            use_boundary_prefetch = (first_layer._hpu_boundary_chunks > 1
                                     and can_pipeline_prefill(hidden_states, first_layer._hpu_boundary_threshold))
        if use_boundary_prefetch:
            prefetched = None
            for layer in islice(self.layers, self.start_layer, self.end_layer):
                hidden_states, residual, prefetched = layer(positions=positions,
                                                            hidden_states=hidden_states,
                                                            residual=residual,
                                                            prefetched=prefetched,
                                                            return_prefetched=True)
            used_grouped_final_norm = False
        elif can_use_hpu_qwen3_layer_groups(
                layer_groups,
                self.aux_hidden_state_layers,
                attn_metadata,
                hidden_states.shape[0],
        ):
            for layer_group in layer_groups:
                hidden_states, residual = layer_group(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
            used_grouped_final_norm = getattr(self, "_hpu_compiled_groups_include_final_norm", False)
        else:
            used_grouped_final_norm = False
            for layer_idx, layer in enumerate(
                    islice(self.layers, self.start_layer, self.end_layer),
                    start=self.start_layer,
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
                self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        if not used_grouped_final_norm:
            hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


# Save original forwards before patching
_orig_qwen3next_attention_forward = Qwen3NextAttention.forward


# ====================================================================
# Qwen3NextAttention.forward  (full-attention layers)
# Patch any 3D layout (decode or bucketed prefill with BS > 1):
#   hidden_states: [B, L, H] -> returns [B, L, H_out]
#
# Return-based since upstream vLLM #46998 (300e33797f) dropped the
# ``output`` in-place buffer; caller now does
#   hidden_states = self.self_attn(hidden_states=..., positions=...)
# ====================================================================
def _hpu_qwen3next_attention_forward(self, positions, hidden_states, return_core=False, projected_input=None):

    # Patch any 3D layout (BS > 1):
    #   Decode:  hidden_states [B, 1, H]
    #   Prefill: hidden_states [B, L, H]
    #
    # Upstream forward assumes 2D (tokens, dim) for attn_output but
    # preserves 3D for gate when hidden_states is 3D, causing a shape
    # mismatch in `attn_output * gate`.  We flatten both to 2D.
    is_3d = (hidden_states is not None and hidden_states.dim() == 3)
    if not is_3d:
        if return_core or projected_input is not None:
            if projected_input is None:
                qkv, _ = self.qkv_proj(hidden_states)
            else:
                qkv = projected_input[0]
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            attn_output = self.attn(q, k, v)
            if gate is not None:
                attn_output = attn_output * torch.sigmoid(gate)
            if return_core:
                return attn_output
            return self.o_proj(attn_output)[0]
        return _orig_qwen3next_attention_forward(self, positions, hidden_states)

    orig_shape = hidden_states.shape

    if projected_input is None:
        qkv, _ = self.qkv_proj(hidden_states)
    else:
        qkv = projected_input[0].reshape(*orig_shape[:-1], -1)

    gate = None
    if self.attn_output_gate:
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        gate_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*gate_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)

        q = q.reshape(*gate_shape, -1)
        gate = gate.reshape(*gate_shape, -1)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
    k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

    q, k = self.rotary_emb(positions, q, k)

    # Normalize attention output to 2D token-major layout.
    attn_output = self.attn(q, k, v)
    attn_output_2d = attn_output.view(-1, attn_output.shape[-1])

    if self.attn_output_gate:
        assert gate is not None
        gate_2d = torch.sigmoid(gate).view(-1, gate.shape[-1])
        attn_output_2d = attn_output_2d * gate_2d

    if return_core:
        return attn_output_2d
    proj_out, _ = self.o_proj(attn_output_2d)

    # Restore caller's original 3-D layout [B, L, H_out] so the residual
    # add in the decoder layer stays shape-consistent.
    return proj_out.view(*orig_shape[:-1], proj_out.shape[-1])


# ====================================================================
# 2. Qwen3NextSparseMoeBlock.forward  (MoE layers)
#    Upstream assumes 2-D input (num_tokens, hidden_dim).  On HPU the
#    hidden_states may arrive as 3-D [B, seq, H] during decode, so we
#    reshape to 2-D first and restore the original shape on output.
# ====================================================================
def _hpu_qwen3next_sparse_moe_forward(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    # Upstream removed the MoERunner.is_internal_router property (vllm PR
    # #51838); its body was simply `self.gate is not None`. Inline that check
    # so the runner-holds-the-gate path still routes through the internal gate.
    if getattr(self.experts, "gate", None) is not None:
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
    else:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]

    return final_hidden_states.reshape(orig_shape)


# ====================================================================
# Apply residual=zeros fix to Qwen3.5/Qwen3Next models
# ====================================================================
def apply_hpu_qwen3_residual_fix(model) -> bool:
    """Apply residual=zeros fix to Qwen3.5 MOE or Qwen3Next models.

    Swaps the inner model class from UpstreamQwen3NextModel (or its subclass
    Qwen3_5Model) to HpuQwen3NextModel to eliminate the residual=None type guard.

    Called from apply_model_specific_patches() in hpu_model_runner.py.

    Returns True if the fix was applied, False otherwise.
    """
    # Import here to avoid circular imports
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    # Handle Qwen3_5MoeForConditionalGeneration -> language_model.model
    # or Qwen3_5MoeForCausalLM -> model
    inner_model = None

    # Check for ConditionalGeneration (has language_model)
    if hasattr(model, 'language_model'):
        lm = model.language_model
        if hasattr(lm, 'model'):
            inner_model = lm.model
    # Check for CausalLM (has model directly)
    elif hasattr(model, 'model'):
        inner_model = model.model

    if inner_model is None:
        return False

    # Check if it's a Qwen3_5Model or Qwen3NextModel that needs the fix
    if isinstance(inner_model, Qwen3_5Model):
        # Qwen3_5Model extends Qwen3NextModel, so the HpuQwen3NextModel.forward
        # will work. We swap the __class__ to get the residual=zeros behavior.
        inner_model.__class__ = HpuQwen3NextModel
        return True
    elif isinstance(inner_model, UpstreamQwen3NextModel):
        inner_model.__class__ = HpuQwen3NextModel
        return True

    return False


# ====================================================================
# Apply all patches
# ====================================================================
Qwen3NextAttention.forward = _hpu_qwen3next_attention_forward
Qwen3NextSparseMoeBlock.forward = _hpu_qwen3next_sparse_moe_forward
