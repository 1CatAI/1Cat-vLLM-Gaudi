import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.forward_context import get_forward_context

import vllm_gaudi.envs as gaudi_envs
from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_fn_token_major,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
    hpu_fused_rmsnorm_gated,
    resolve_hpu_gdn_chunk_size,
    resolve_hpu_gdn_compact_repeated_kkt,
    resolve_hpu_gdn_compact_repeated_local_attn,
    resolve_hpu_gdn_compiled_qk_l2norm,
    resolve_hpu_gdn_fused_rmsnorm_gated,
    resolve_hpu_gdn_fused_state_matmul,
    resolve_hpu_gdn_neumann_iters,
    resolve_hpu_gdn_recursive_solver_base,
)
from vllm_gaudi.ops.qwen38_native_qk import (
    load_qwen38_native_qk_prep,
    qwen38_native_qk_prep,
    validate_qwen38_native_qk_shape,
)
from vllm_gaudi.ops.flashinfer_gaudi_adapter import (
    maybe_run_gdn_decode_packed,
    maybe_run_gdn_fused_decode_step,
    maybe_run_gdn_prefill,
)


def _save_ssm_state(core_attn_out, final_state, ssm_state, state_indices):
    """Persist GDN final_state into ssm_state cache for chunked prefill.

    Must be @torch._dynamo.disable because HPU torch.compile silently
    drops in-place index_copy_ to aliased state tensors.  Returns
    core_attn_out as a pass-through so the compiled graph consumes
    the call — HPU drops dynamo-disabled calls whose results are unused.
    """
    state_indices = state_indices.reshape(-1).to(device=ssm_state.device, dtype=torch.long)
    if state_indices.numel() == 1:
        # A real single-request prefill always owns its only state slot. Keep
        # this hot path to one index operation per GDN layer; materializing
        # nonzero(valid) here adds a device/host synchronization.
        safe_si = torch.remainder(state_indices, ssm_state.shape[0])
        ssm_state.index_copy_(0, safe_si, final_state.to(device=ssm_state.device, dtype=ssm_state.dtype))
        return core_attn_out
    valid = (state_indices >= 0) & (state_indices < ssm_state.shape[0])
    valid_positions = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if valid_positions.numel() > 0:
        safe_si = state_indices.index_select(0, valid_positions)
        state_rows = final_state.index_select(0, valid_positions).to(device=ssm_state.device, dtype=ssm_state.dtype)
        ssm_state.index_copy_(0, safe_si, state_rows)
    return core_attn_out


class HPUGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache_group_idx: set later by model runner for hybrid cache
        # lookup.  Stored as tensor so torch.compile treats it as dynamic.
        self.cache_group_idx = None
        self.compact_state_group_offset = None
        self.compact_state_group_count = None

        self.mamba_chunk_size, _ = resolve_hpu_gdn_chunk_size(self.model_config)
        self.gdn_fused_state_matmul = resolve_hpu_gdn_fused_state_matmul()
        self.gdn_neumann_iters = resolve_hpu_gdn_neumann_iters()
        self.gdn_recursive_solver_base = resolve_hpu_gdn_recursive_solver_base()
        self.gdn_compact_repeated_kkt = resolve_hpu_gdn_compact_repeated_kkt()
        self.gdn_compact_repeated_local_attn = (
            resolve_hpu_gdn_compact_repeated_local_attn()
        )
        self.gdn_compiled_qk_l2norm = resolve_hpu_gdn_compiled_qk_l2norm()
        self.gdn_fused_rmsnorm_gated = resolve_hpu_gdn_fused_rmsnorm_gated()

        self.qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        self.z_size = self.value_dim // self.tp_size
        self.gdn_native_qk_prep = load_qwen38_native_qk_prep()
        self.gdn_compact_qk_input = gaudi_envs.VLLM_GDN_QWEN38_COMPACT_QK
        self.gdn_native_compact_kkt = (
            gaudi_envs.VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT
        )
        self.gdn_compact_qk_factor_gate = (
            gaudi_envs.VLLM_GDN_COMPACT_QK_FACTOR_GATE
        )
        if self.gdn_compact_qk_factor_gate and not self.gdn_compact_qk_input:
            raise ValueError(
                "VLLM_GDN_COMPACT_QK_FACTOR_GATE requires compact Q/K."
            )
        if self.gdn_compact_qk_input and not self.gdn_native_qk_prep:
            raise ValueError(
                "VLLM_GDN_QWEN38_COMPACT_QK requires native Q/K preparation."
            )
        if self.gdn_native_compact_kkt and not self.gdn_compact_qk_input:
            raise ValueError(
                "VLLM_GDN_QWEN38_NATIVE_COMPACT_KKT requires compact Q/K."
            )
        self.gdn_token_major_causal_conv1d = (
            gaudi_envs.VLLM_GDN_TOKEN_MAJOR_CAUSAL_CONV1D
        )
        self.gdn_native_qk_head_repeat = None
        if self.gdn_native_qk_prep:
            self.gdn_native_qk_head_repeat = validate_qwen38_native_qk_shape(
                tp_size=self.tp_size,
                qkv_width=self.qkv_size,
                key_width=self.key_dim // self.tp_size,
                value_width=self.value_dim // self.tp_size,
                key_head_dim=self.head_k_dim,
                value_head_dim=self.head_v_dim,
            )

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Pure-torch rearrange – avoids einops graph breaks on HPU."""
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query = query.reshape(1, query.size(0), -1, self.head_k_dim).contiguous()
        key = key.reshape(1, key.size(0), -1, self.head_k_dim).contiguous()
        value = value.reshape(1, value.size(0), -1, self.head_v_dim).contiguous()
        return query, key, value

    def _resolve_state_indices(self, attn_metadata, attribute):
        """Resolve one state-index tensor, handling 2-D cache groups."""
        indices = getattr(attn_metadata, attribute, None)
        if indices is not None and indices.dim() > 1:
            cg = self.cache_group_idx
            assert cg is not None
            indices = indices.index_select(0, cg.view(1)).squeeze(0)
        return indices

    def _extract_metadata(self, num_tokens):
        """Extract forward-context metadata into plain tensors.

        Dynamo graph-breaks naturally on ``get_forward_context()``; no
        ``@dynamo.disable`` needed.
        """
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return (False, None, None, None, None, None, None, None, 0, 0, 0, 0, None, False)

        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        load_state_indices = self._resolve_state_indices(attn_metadata, "load_indices_tensor")
        store_state_indices = self._resolve_state_indices(attn_metadata, "store_indices_tensor")
        if not getattr(self.cache_config, "enable_prefix_caching", False):
            # The scheduler constructs equivalent load/store tensors in this
            # mode. Preserve object identity so single-index native kernels
            # can be selected without a device-side equality synchronization.
            store_state_indices = load_state_indices
        elif store_state_indices is None:
            store_state_indices = load_state_indices

        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]

        query_start_loc = attn_metadata.query_start_loc_p
        has_initial_state = getattr(attn_metadata, "has_initial_states_p", None)
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)
        direct_gdn_state = bool(getattr(attn_metadata, "direct_gdn_state", False))

        if not is_prompt:
            num_decodes = (load_state_indices.numel() if load_state_indices is not None else
                           (query_start_loc.numel() - 1 if query_start_loc is not None else num_tokens))
        else:
            num_decodes = 0

        mamba_block_size = (self.cache_config.mamba_block_size if is_prompt else 0)

        # Prefill-specific metadata (Python ints for torch.compile)
        prefill_num_seqs = 0
        prefill_seq_len = 0
        initial_state = None
        if is_prompt and load_state_indices is not None:
            prefill_num_seqs = int(load_state_indices.numel())
            prefill_seq_len = (num_tokens // prefill_num_seqs if prefill_num_seqs > 0 else 0)
            if prefill_num_seqs == 1:
                # Avoid three validity-mask kernels in every GDN layer for
                # the latency-critical one-request prefill path.
                initial_state = ssm_state[load_state_indices].contiguous()
            else:
                safe_load_indices = torch.where(
                    (load_state_indices >= 0) & (load_state_indices < ssm_state.shape[0]),
                    load_state_indices,
                    torch.zeros_like(load_state_indices),
                ).long()
                initial_state = ssm_state.index_select(0, safe_load_indices).contiguous()
            if has_initial_state is not None:
                mask = has_initial_state.bool().view(-1, 1, 1, 1)
                initial_state = torch.where(
                    mask,
                    initial_state,
                    torch.zeros_like(initial_state),
                )

        return (is_prompt, conv_state, ssm_state, load_state_indices, store_state_indices, query_start_loc,
                has_initial_state, padding_mask_flat, num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len,
                initial_state, direct_gdn_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """HPU compile-friendly GDN forward.

        Bypasses the upstream ``gdn_attention_core`` custom-op and
        drives the HPU conv1d + GDN kernels directly with
        ``HPUAttentionMetadataV1``.

        Return-based since upstream vLLM #46998 (300e33797f) dropped the
        ``output`` in-place buffer; caller now does
        ``hidden_states = self.linear_attn(hidden_states=...)``.
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # === Metadata extraction (natural graph break) ===============
        (is_prompt, conv_state, ssm_state, load_state_indices, store_state_indices, query_start_loc, has_initial_state,
         padding_mask_flat, num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state,
         direct_gdn_state) = self._extract_metadata(num_tokens)

        # === Part 1: Input Projection ================================
        if hasattr(self, 'in_proj_qkv'):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
                # Pure-torch flatten instead of einops rearrange (graph breaks)
                query = query.reshape(query.size(0), -1)
                key = key.reshape(key.size(0), -1)
                value = value.reshape(value.size(0), -1)
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights already in [q, k, v, z] and [b, a] order
                mixed_qkv, z = mixed_qkvz.split([self.qkv_size, self.z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if conv_state is None:
            # No attn_metadata — skip core attention (profile run)
            pass
        elif is_prompt:
            # === Part 2a: Prefill ====================================
            if (padding_mask_flat is not None and padding_mask_flat.numel() == num_tokens):
                token_mask_flat = padding_mask_flat.view(-1, 1).to(dtype=mixed_qkv.dtype)
                mixed_qkv = mixed_qkv * token_mask_flat
            else:
                token_mask_flat = None

            g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)

            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if self.gdn_token_major_causal_conv1d:
                mixed_qkv_conv = hpu_causal_conv1d_fn_token_major(
                    x=mixed_qkv,
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=load_state_indices,
                    query_start_loc=query_start_loc,
                    block_size_to_align=mamba_block_size,
                    is_prompt=True,
                )
            else:
                mixed_qkv_conv = hpu_causal_conv1d_fn(
                    x=mixed_qkv.transpose(0, 1),
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    conv_states=conv_state,
                    has_initial_state=has_initial_state,
                    cache_indices=load_state_indices,
                    block_idx_first_scheduled_token=None,
                    block_idx_last_scheduled_token=None,
                    initial_state_idx=None,
                    query_start_loc=query_start_loc,
                    block_size_to_align=mamba_block_size,
                    num_computed_tokens=None,
                    metadata=None,
                    is_prompt=True,
                ).transpose(0, 1)

            if token_mask_flat is not None:
                mixed_qkv_conv = mixed_qkv_conv * token_mask_flat

            if self.gdn_native_qk_prep:
                query, key = qwen38_native_qk_prep(mixed_qkv_conv)
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value_offset = 2 * (self.key_dim // self.tp_size)
                value = mixed_qkv_conv.narrow(
                    -1,
                    value_offset,
                    self.value_dim // self.tp_size,
                ).reshape(
                    1,
                    mixed_qkv_conv.size(0),
                    self.num_v_heads // self.tp_size,
                    self.head_v_dim,
                ).contiguous()
                use_qk_l2norm_in_kernel = False
            else:
                query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)
                use_qk_l2norm_in_kernel = True

            if token_mask_flat is not None:
                token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
                g = g * token_mask_h
                beta = beta * token_mask_h

            flashinfer_prefill_result = maybe_run_gdn_prefill(
                q=query,
                k=key,
                v=value,
                log_decay=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                chunk_size=self.mamba_chunk_size,
                prefill_num_seqs=prefill_num_seqs,
                prefill_seq_len=prefill_seq_len,
                scale=self.head_k_dim**-0.5,
            )
            if flashinfer_prefill_result is not None:
                core_attn_out_result, final_state = flashinfer_prefill_result
            else:
                core_attn_out_result, final_state = hpu_chunk_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    chunk_size=self.mamba_chunk_size,
                    prefill_num_seqs=prefill_num_seqs,
                    prefill_seq_len=prefill_seq_len,
                    neumann_iters=self.gdn_neumann_iters,
                    fused_state_matmul=self.gdn_fused_state_matmul,
                    recursive_solver_base=self.gdn_recursive_solver_base,
                    compact_repeated_kkt=self.gdn_compact_repeated_kkt,
                    compile_qk_l2norm=self.gdn_compiled_qk_l2norm,
                    qk_head_repeat_override=self.gdn_native_qk_head_repeat,
                    compact_repeated_local_attn=self.gdn_compact_repeated_local_attn,
                    preserve_compact_qk=self.gdn_compact_qk_input,
                    compact_qk_factor_gate=self.gdn_compact_qk_factor_gate,
                )
            assert final_state is not None
            # State save in dynamo-disabled wrapper — index_copy_ is
            # silently dropped by HPU torch.compile on aliased tensors.
            core_attn_out_result = _save_ssm_state(
                core_attn_out_result,
                final_state,
                ssm_state,
                store_state_indices,
            )

            non_spec_out = core_attn_out_result.squeeze(0)
            core_attn_out[:non_spec_out.shape[0]] = non_spec_out

        else:
            # === Part 2b: Decode =====================================
            selected_conv_state = conv_state
            direct_conv_state = False
            if (direct_gdn_state and self.compact_state_group_count is not None and self.compact_state_group_count > 0
                    and self.compact_state_group_offset is not None):
                group_span = (conv_state.shape[0] - 2) // self.compact_state_group_count
                if num_decodes <= group_span:
                    state_start = self.compact_state_group_offset * group_span + 1
                    selected_conv_state = conv_state.narrow(0, state_start, num_decodes)
                    direct_conv_state = True
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0),
                self.conv1d.weight.size(2),
            )
            flashinfer_result = maybe_run_gdn_fused_decode_step(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                conv_state=selected_conv_state,
                conv_weight=conv_weights,
                conv_bias=self.conv1d.bias,
                ssm_state=ssm_state,
                load_state_indices=load_state_indices,
                direct_conv_state=direct_conv_state,
                direct_gdn_state=direct_gdn_state,
                direct_state_group_count=self.compact_state_group_count,
                direct_state_group_offset=self.compact_state_group_offset,
                scale=self.head_k_dim**-0.5,
            )
            if flashinfer_result is None:
                g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
                mixed_qkv_conv = hpu_causal_conv1d_update(
                    x=mixed_qkv,
                    conv_state=selected_conv_state,
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=(None if direct_conv_state else (
                        load_state_indices[:num_decodes] if load_state_indices is not None else load_state_indices)),
                    block_idx_last_scheduled_token=None,
                    initial_state_idx=None,
                    query_start_loc=query_start_loc,
                    validate_data=False,
                    direct_state_layout=direct_conv_state,
                )
                flashinfer_result = maybe_run_gdn_decode_packed(
                    mixed_qkv=mixed_qkv_conv,
                    log_decay=g,
                    beta=beta,
                    state_pool=ssm_state,
                    load_state_indices=load_state_indices,
                    store_state_indices=store_state_indices,
                    use_qk_l2norm=True,
                    scale=self.head_k_dim**-0.5,
                    direct_state_layout=direct_gdn_state,
                    direct_state_group_count=self.compact_state_group_count,
                    direct_state_group_offset=self.compact_state_group_offset,
                )
                if flashinfer_result is None:
                    query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)
                    core_attn_out_result, _ = hpu_fused_recurrent_gated_delta_rule(
                        q=query,
                        k=key,
                        v=value,
                        g=g,
                        beta=beta,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=(query_start_loc[:num_decodes + 1] if query_start_loc is not None else None),
                        ssm_state_indices=load_state_indices,
                        use_qk_l2norm_in_kernel=True,
                    )
            if flashinfer_result is not None:
                core_attn_out_result, _ = flashinfer_result

            non_spec_out = core_attn_out_result.squeeze(0)
            if non_spec_out.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(non_spec_out)
            else:
                n = min(non_spec_out.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = non_spec_out[:n]

        # === Part 3: Output Projection ===============================
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if is_prompt and self.gdn_fused_rmsnorm_gated:
            core_attn_out = hpu_fused_rmsnorm_gated(
                core_attn_out,
                z,
                self.norm.weight,
                self.norm.eps,
                self.norm.activation,
            )
        else:
            core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)

        output, _ = self.out_proj(core_attn_out)
        # Restore caller's original layout (2-D flat or 3-D [B, L, H]) so
        # the residual add in the decoder layer stays shape-consistent.
        return output.view(orig_shape)


# Replace the class in the upstream modules so that both Qwen3-Next and
# Qwen3.5 model definitions instantiate HPUGatedDeltaNetAttention.
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_module  # noqa: E402
import vllm.model_executor.models.qwen3_next as _qwen3_next_module  # noqa: E402
import vllm.model_executor.models.qwen3_5 as _qwen3_5_module  # noqa: E402

_gdn_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_next_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
_qwen3_5_module.QwenGatedDeltaNetAttention = HPUGatedDeltaNetAttention
