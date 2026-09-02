import os

import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.forward_context import get_forward_context

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
)


# Import only for enabled runs. The runtime keeps small hybrid decode buckets
# on the vendor graph and selects Triton only where the composite performance
# gate has cleared; strict mode remains the fail-closed A/B path.
_triton_gaudi_mode = os.environ.get("VLLM_HPU_TRITON_MODE", "off").strip().lower()
if _triton_gaudi_mode in ("hybrid", "strict"):
    from vllm_gaudi.ops.triton_gaudi import (
        gdn_decode_conv_split_packed as _triton_gdn_decode_conv_packed,
        gdn_decode_packed as _triton_gdn_decode_packed,
    )
else:
    _triton_gdn_decode_conv_packed = None
    _triton_gdn_decode_packed = None


def _try_triton_gdn_decode_conv_packed(
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    mixed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor | None,
    conv_weight_t: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run the performance-gated split width-4 conv + GDN fast path."""
    if _triton_gdn_decode_conv_packed is None:
        return None
    if state_indices is None or conv_weight_t is None:
        if _triton_gaudi_mode == "strict":
            missing = "state_indices" if state_indices is None else "transposed conv weight"
            raise RuntimeError(
                f"Gaudi Triton strict fused GDN decode requires {missing}")
        return None
    return _triton_gdn_decode_conv_packed(
        conv_state,
        ssm_state,
        mixed_qkv.contiguous(),
        gate_a.contiguous(),
        gate_b.contiguous(),
        a_log.contiguous(),
        dt_bias.contiguous(),
        state_indices.contiguous(),
        conv_weight_t,
    )


def _try_triton_gdn_decode_packed(
    ssm_state: torch.Tensor,
    mixed_qkv_conv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run the performance-gated packed Qwen3.5 decode candidate.

    Keeping this boundary immediately after causal-conv lets Dynamo place the
    vendor conv and the state-mutating Triton op in one HPU graph. The runtime
    validator owns the canonical TP1/single-token specialization checks.
    """
    if _triton_gdn_decode_packed is None:
        return None
    if state_indices is None:
        if _triton_gaudi_mode == "strict":
            raise RuntimeError(
                "Gaudi Triton strict GDN decode requires state_indices metadata")
        return None
    return _triton_gdn_decode_packed(
        ssm_state,
        mixed_qkv_conv.contiguous(),
        gate_a.contiguous(),
        gate_b.contiguous(),
        a_log.contiguous(),
        dt_bias.contiguous(),
        state_indices.contiguous(),
    )


def _save_ssm_state(core_attn_out, final_state, ssm_state, state_indices):
    """Persist GDN final_state into ssm_state cache for chunked prefill.

    Must be @torch._dynamo.disable because HPU torch.compile silently
    drops in-place index_copy_ to aliased state tensors.  Returns
    core_attn_out as a pass-through so the compiled graph consumes
    the call — HPU drops dynamo-disabled calls whose results are unused.
    """
    safe_si = torch.remainder(state_indices, ssm_state.shape[0]).long()
    ssm_state.index_copy_(0, safe_si, final_state.to(device=ssm_state.device, dtype=ssm_state.dtype))
    return core_attn_out


class HPUGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cache_group_idx: set later by model runner for hybrid cache
        # lookup.  Stored as tensor so torch.compile treats it as dynamic.
        self.cache_group_idx = None

        # mamba_chunk_size: use explicit config value or default to 128
        # for HPU bucket alignment.
        hf_text_config = getattr(self.model_config, "hf_text_config", None)
        has_explicit = (hf_text_config is not None and (getattr(hf_text_config, "mamba_chunk_size", None) is not None
                                                        or getattr(hf_text_config, "chunk_size", None) is not None))
        self.mamba_chunk_size = (self.model_config.get_mamba_chunk_size() if has_explicit else 128)

        self.qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        self.z_size = self.value_dim // self.tp_size

        # The split TPC kernels vector-load one convolution tap across channels.
        # Materialize that tap-major layout once while the checkpoint loader
        # writes the canonical [channels, 1, width] parameter.
        self._triton_conv_weight_ready = False
        self._triton_dt_bias_ready = False
        if _triton_gaudi_mode in ("hybrid", "strict"):
            conv_weight = self.conv1d.weight
            self.register_buffer(
                "_triton_conv_weight_t",
                conv_weight.new_empty((self.conv_kernel_size, conv_weight.size(0))),
                persistent=False,
            )
            self.register_buffer(
                "_triton_dt_bias_f32",
                self.dt_bias.new_empty(self.dt_bias.shape, dtype=torch.float32),
                persistent=False,
            )
            original_weight_loader = conv_weight.weight_loader
            original_dt_bias_loader = self.dt_bias.weight_loader

            def load_conv_weight_and_transpose(param, loaded_weight):
                result = original_weight_loader(param, loaded_weight)
                with torch.no_grad():
                    self._triton_conv_weight_t.copy_(
                        param.view(param.size(0), param.size(2)).transpose(0, 1))
                self._triton_conv_weight_ready = True
                return result

            conv_weight.weight_loader = load_conv_weight_and_transpose

            def load_dt_bias_as_fp32(param, loaded_weight):
                result = original_dt_bias_loader(param, loaded_weight)
                with torch.no_grad():
                    self._triton_dt_bias_f32.copy_(param.float())
                self._triton_dt_bias_ready = True
                return result

            self.dt_bias.weight_loader = load_dt_bias_as_fp32

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

    def _resolve_state_indices(self, attn_metadata):
        """Resolve load_indices_tensor, handling 2-D cache-group case.

        For Qwen 3.5 (GDN), load and store indices are identical
        so using load_indices_tensor is sufficient.
        """
        indices = attn_metadata.load_indices_tensor
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
            return (False, None, None, None, None, None, None, 0, 0, 0, 0, None)

        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        state_indices = self._resolve_state_indices(attn_metadata)

        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]

        query_start_loc = attn_metadata.query_start_loc_p
        has_initial_state = getattr(attn_metadata, "has_initial_states_p", None)
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)

        if not is_prompt:
            num_decodes = (state_indices.numel() if state_indices is not None else
                           (query_start_loc.numel() - 1 if query_start_loc is not None else num_tokens))
        else:
            num_decodes = 0

        mamba_block_size = (self.cache_config.mamba_block_size if is_prompt else 0)

        # Prefill-specific metadata (Python ints for torch.compile)
        prefill_num_seqs = 0
        prefill_seq_len = 0
        initial_state = None
        if is_prompt and state_indices is not None:
            prefill_num_seqs = int(state_indices.numel())
            prefill_seq_len = (num_tokens // prefill_num_seqs if prefill_num_seqs > 0 else 0)
            initial_state = ssm_state[state_indices].contiguous()
            if has_initial_state is not None:
                # Avoid scatter_nd from boolean indexing
                mask = has_initial_state.bool().view(-1, 1, 1, 1).to(initial_state.dtype)
                initial_state = initial_state * mask

        return (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
                num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len, initial_state)

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
        (is_prompt, conv_state, ssm_state, state_indices, query_start_loc, has_initial_state, padding_mask_flat,
         num_decodes, mamba_block_size, prefill_num_seqs, prefill_seq_len,
         initial_state) = self._extract_metadata(num_tokens)

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
            mixed_qkv_conv = hpu_causal_conv1d_fn(
                x=mixed_qkv.transpose(0, 1),
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=state_indices,
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

            query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

            if token_mask_flat is not None:
                token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
                g = g * token_mask_h
                beta = beta * token_mask_h

            core_attn_out_result, final_state = hpu_chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                chunk_size=self.mamba_chunk_size,
                prefill_num_seqs=prefill_num_seqs,
                prefill_seq_len=prefill_seq_len,
            )
            # State save in dynamo-disabled wrapper — index_copy_ is
            # silently dropped by HPU torch.compile on aliased tensors.
            core_attn_out_result = _save_ssm_state(
                core_attn_out_result,
                final_state,
                ssm_state,
                state_indices,
            )

            non_spec_out = core_attn_out_result.squeeze(0)
            core_attn_out[:non_spec_out.shape[0]] = non_spec_out

        else:
            # === Part 2b: Decode =====================================
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            triton_out = _try_triton_gdn_decode_conv_packed(
                conv_state,
                ssm_state,
                mixed_qkv,
                a,
                b,
                self.A_log,
                (self._triton_dt_bias_f32
                 if self._triton_dt_bias_ready else self.dt_bias),
                state_indices,
                (self._triton_conv_weight_t
                 if self._triton_conv_weight_ready else None),
            )
            if triton_out is None:
                mixed_qkv_conv = hpu_causal_conv1d_update(
                    x=mixed_qkv,
                    conv_state=conv_state,
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    conv_state_indices=(
                        state_indices[:num_decodes]
                        if state_indices is not None else state_indices),
                    block_idx_last_scheduled_token=None,
                    initial_state_idx=None,
                    query_start_loc=query_start_loc,
                    validate_data=False,
                )
                triton_out = _try_triton_gdn_decode_packed(
                    ssm_state,
                    mixed_qkv_conv,
                    a,
                    b,
                    self.A_log,
                    (self._triton_dt_bias_f32
                     if self._triton_dt_bias_ready else self.dt_bias),
                    state_indices,
                )
            if triton_out is None:
                g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
                query, key, value = self.rearrange_mixed_qkv(mixed_qkv_conv)

                core_attn_out_result, _ = \
                    hpu_fused_recurrent_gated_delta_rule(
                        q=query, k=key, v=value, g=g, beta=beta,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=(
                            query_start_loc[:num_decodes + 1]
                            if query_start_loc is not None else None),
                        ssm_state_indices=state_indices,
                        use_qk_l2norm_in_kernel=True,
                    )
                non_spec_out = core_attn_out_result.squeeze(0)
            else:
                non_spec_out = triton_out
            if non_spec_out.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(non_spec_out)
            else:
                n = min(non_spec_out.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = non_spec_out[:n]

        # === Part 3: Output Projection ===============================
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
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
