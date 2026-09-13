# SPDX-License-Identifier: Apache-2.0
"""DFlash2 proposer for the vLLM-Gaudi V1 model runner.

The CUDA implementation prepares inputs and walks the selector lattice with
Triton. Gaudi has no Triton runtime, so this proposer keeps the same model and
KV-cache contracts while using static, request-major HPU tensors and the
FlashInfer-Gaudi primitives.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch

from flashinfer_gaudi.dflash2 import build_attention_bias
from vllm.config import replace
from vllm_gaudi.utils import async_h2d_copy
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm_gaudi.v1.spec_decode.hpu_eagle import HpuEagleProposer


class HpuDFlash2Proposer(HpuEagleProposer):
    """One-pass, greedy DFlash2 proposer specialized for Gaudi."""

    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device, runner)
        # DFlash embeds its bonus and mask tokens. It does not consume the
        # EAGLE-only persistent mask-hidden buffer during model loading.
        self.parallel_drafting_hidden_state_tensor = None
        draft_hf_config = self.draft_model_config.hf_config
        dflash_config = draft_hf_config.dflash_config
        self.num_query_per_req = 1 + self.num_speculative_tokens
        self.mask_token_id = int(dflash_config["mask_token_id"])
        self.selector_top_k = int(dflash_config["selector_top_k"])
        self.use_aux_hidden_state = bool(
            dflash_config.get(
                "use_aux_hidden_state",
                getattr(draft_hf_config, "use_aux_hidden_state", True),
            ))
        self.sliding_window = dflash_config.get(
            "swa_window_size",
            getattr(draft_hf_config, "sliding_window", None),
        )
        from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

        self.dflash_causal = not dflash_has_any_non_causal(draft_hf_config)
        self.kv_cache_gid = -1
        self.block_size = -1
        self._model_api_cache: Any | None = None

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        draft_hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(draft_hf_config, "dflash_config", None) or {}
        return bool(dflash_config.get(
            "use_aux_hidden_state",
            getattr(draft_hf_config, "use_aux_hidden_state", True),
        ))

    def _create_draft_vllm_config(self):
        """Mirror upstream DFlash attention configuration for the HPU draft."""
        base = super()._create_draft_vllm_config()
        architecture = base.model_config.model_arch_config
        if architecture.is_mm_prefix_lm:
            base.model_config.model_arch_config = replace(architecture, is_mm_prefix_lm=False)
        return replace(
            base,
            attention_config=replace(
                base.attention_config,
                use_non_causal=not self.dflash_causal,
            ),
        )

    def load_model(self, target_model: torch.nn.Module) -> None:
        """Match the target RoPE convention before constructing the draft."""
        from vllm.model_executor.models.qwen3_dflash import dflash_target_rope_is_neox_style

        is_neox_style = dflash_target_rope_is_neox_style(target_model)
        if is_neox_style is not None:
            self.draft_model_config.hf_config.is_neox_style = is_neox_style
        super().load_model(target_model)

    def get_hpu_adapter_config(self):
        """Use the draft model config when the HPU adapter builds metadata."""
        config = copy.copy(self._create_draft_vllm_config())
        config.model_config = self.draft_model_config
        return config

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        """Resolve the draft KV group without creating CUDA metadata builders."""
        group_ids = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                group_ids.append(group_id)
        if len(group_ids) != 1:
            raise ValueError("HPU DFlash2 requires all draft attention layers in one KV "
                             f"cache group, found {group_ids}.")
        self.kv_cache_gid = group_ids[0]
        if kernel_block_sizes is not None and self.kv_cache_gid < len(kernel_block_sizes):
            self.block_size = int(kernel_block_sizes[self.kv_cache_gid])
        else:
            group_spec = kv_cache_config.kv_cache_groups[self.kv_cache_gid].kv_cache_spec
            self.block_size = int(group_spec.block_size)

    @staticmethod
    def _flatten_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.reshape(-1, hidden_states.shape[-1])

    def combine_context_hidden_states(
        self,
        hidden_states: torch.Tensor,
        aux_hidden_states: Sequence[torch.Tensor] | None,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        if aux_hidden_states:
            selected = [self._flatten_hidden(states).index_select(0, source_indices) for states in aux_hidden_states]
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = self._flatten_hidden(hidden_states).index_select(0, source_indices)
        return self._model_api().combine_hidden_states(hidden_states)

    def combine_context_hidden_states_fixed(
        self,
        hidden_states: torch.Tensor,
        aux_hidden_states: Sequence[torch.Tensor] | None,
        num_context_tokens: int,
    ) -> torch.Tensor:
        """Combine the full fixed-width target block without host compaction."""
        if aux_hidden_states:
            selected = [self._flatten_hidden(states)[:num_context_tokens] for states in aux_hidden_states]
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = self._flatten_hidden(hidden_states)[:num_context_tokens]
        return self._model_api().combine_hidden_states(hidden_states)

    def _model_api(self) -> Any:
        """Find the HPU adapter through eager or HPUGraph wrappers."""
        if self._model_api_cache is not None:
            return self._model_api_cache
        model = self.model
        visited: set[int] = set()
        while id(model) not in visited:
            visited.add(id(model))
            required = (
                "combine_hidden_states",
                "precompute_and_store_context_kv",
                "compute_candidates",
                "prepare_dflash2_inputs",
                "score_dflash2_candidates",
                "select_dflash2_candidates",
            )
            if all(hasattr(model, name) for name in required):
                self._model_api_cache = model
                return model
            for attribute in ("module", "model", "_model"):
                child = getattr(model, attribute, None)
                if child is not None and child is not model:
                    model = child
                    break
            else:
                break
        raise RuntimeError("Could not resolve the DFlash2 model API through the HPU wrapper")

    @staticmethod
    def _resolved_block_id(block_table: torch.Tensor, row: int, logical_block: int, runner) -> int:
        if row < 0 or row >= block_table.shape[0] or logical_block < 0 or logical_block >= block_table.shape[1]:
            return int(runner._PAD_BLOCK_ID)
        block_id = int(block_table[row, logical_block].item())
        if block_id < 0 or block_id == runner._PAD_BLOCK_ID:
            return int(runner._PAD_BLOCK_ID)
        return int(runner._resolve_block(block_id))

    def make_slot_mapping(
        self,
        block_table: torch.Tensor,
        row_indices: Sequence[int],
        positions: Sequence[Sequence[int]],
        runner,
    ) -> torch.Tensor:
        """Build a CPU physical-slot tensor from a draft-group block table."""
        if self.block_size <= 0:
            raise RuntimeError("DFlash2 attention backend has not been initialized")
        pad_slot = int(runner._PAD_SLOT_ID)
        slots: list[list[int]] = []
        for row, row_positions in zip(row_indices, positions):
            request_slots = []
            for position in row_positions:
                logical_block, offset = divmod(int(position), self.block_size)
                block_id = self._resolved_block_id(block_table, int(row), logical_block, runner)
                slot = pad_slot if block_id == runner._PAD_BLOCK_ID else block_id * self.block_size + offset
                request_slots.append(slot)
            slots.append(request_slots)
        return torch.tensor(slots, dtype=torch.int64, device="cpu")

    def store_context(
        self,
        context_states: torch.Tensor,
        context_positions: Sequence[int],
        context_slots: torch.Tensor,
    ) -> None:
        if not context_positions:
            return
        positions = async_h2d_copy(torch.tensor(context_positions, dtype=torch.int64), device=self.device)
        slots = async_h2d_copy(context_slots.reshape(-1), device=self.device)
        self._model_api().precompute_and_store_context_kv(context_states, positions, slots)

    def store_context_device(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slots: torch.Tensor,
    ) -> None:
        """Store a fixed-width context whose rejected rows target pad slots."""
        self._model_api().precompute_and_store_context_kv(context_states, context_positions, context_slots)

    def _build_attention_metadata(
        self,
        query_starts: Sequence[int],
        query_slots: torch.Tensor,
        block_table: torch.Tensor,
        row_indices: Sequence[int],
        runner,
    ) -> HPUAttentionMetadataV1:
        batch_size = len(query_starts)
        query_len = self.num_query_per_req
        window = int(self.sliding_window) if self.sliding_window is not None else None

        first_blocks: list[int] = []
        context_blocks: list[list[int]] = []
        for row, context_end in zip(row_indices, query_starts):
            # FlashAttention's configured window W is represented as W - 1
            # positions on each side of a query.
            first_position = (0 if window is None else max(0, int(context_end) - (window - 1)))
            first_block = first_position // self.block_size
            last_block = (int(context_end) - 1) // self.block_size
            blocks = [
                self._resolved_block_id(block_table, int(row), logical_block, runner)
                for logical_block in range(first_block, last_block + 1)
            ]
            first_blocks.append(first_block)
            context_blocks.append(blocks)

        max_blocks = max((len(blocks) for blocks in context_blocks), default=0)
        padded_blocks = [blocks + [int(runner._PAD_BLOCK_ID)] * (max_blocks - len(blocks)) for blocks in context_blocks]
        past_width = max_blocks * self.block_size
        context_lens: list[int] = []
        for batch_idx, (context_end, first_block) in enumerate(zip(query_starts, first_blocks)):
            frame_base = first_block * self.block_size
            context_len = int(context_end) - frame_base
            context_lens.append(context_len)

        attn_bias = build_attention_bias(
            query_starts=query_starts,
            first_blocks=first_blocks,
            block_size=self.block_size,
            past_width=past_width,
            query_len=query_len,
            window=window,
            causal=self.dflash_causal,
            dtype=self.dtype,
        )
        block_list = None
        if max_blocks:
            block_list = async_h2d_copy(torch.tensor(padded_blocks, dtype=torch.int32).flatten(), device=self.device)
        attn_bias_device = async_h2d_copy(attn_bias, device=self.device)
        return HPUAttentionMetadataV1.make_prefill_metadata(
            attn_bias=attn_bias_device,
            window_attn_bias=attn_bias_device,
            block_list=block_list,
            context_lens_tensor=async_h2d_copy(torch.tensor(context_lens, dtype=torch.int32), device=self.device),
            seq_lens_tensor=async_h2d_copy(
                torch.full((batch_size, ), query_len, dtype=torch.int32),
                device=self.device,
            ),
            slot_mapping=async_h2d_copy(query_slots, device=self.device),
            block_size=self.block_size,
            # The explicit bias carries both causality and window alignment.
            causal=False,
        )

    def prepare_decode_device_inputs(
        self,
        sampled_token_ids: torch.Tensor,
        base_positions: torch.Tensor,
        block_table: torch.Tensor,
        active_mask: torch.Tensor,
        first_blocks: torch.Tensor,
        last_blocks: torch.Tensor,
        runner,
        max_context_blocks: int,
    ) -> tuple[torch.Tensor, ...]:
        """Build the draft bridge tensors without reading accepted lengths."""
        return self._model_api().prepare_dflash2_inputs(
            sampled_token_ids,
            base_positions,
            block_table,
            active_mask,
            first_blocks,
            last_blocks,
            block_size=self.block_size,
            num_query_per_req=self.num_query_per_req,
            mask_token_id=self.mask_token_id,
            pad_block_id=int(runner._PAD_BLOCK_ID),
            pad_slot_id=int(runner._PAD_SLOT_ID),
            max_model_len=int(runner.max_model_len),
            max_context_blocks=max_context_blocks,
            window=int(self.sliding_window) if self.sliding_window is not None else None,
            causal=self.dflash_causal,
            bias_dtype=self.dtype,
        )

    def propose_prepared_query_block(
        self,
        prepared: tuple[torch.Tensor, ...],
        runner,
        real_batch_size: int,
    ) -> torch.Tensor:
        """Run the fixed draft block directly from device-prepared metadata."""
        (
            _accepted_counts,
            _context_positions,
            _context_slots,
            input_ids,
            query_positions,
            query_slots,
            block_list,
            context_lens,
            attn_bias,
        ) = prepared
        batch_size = input_ids.shape[0]
        metadata = HPUAttentionMetadataV1.make_prefill_metadata(
            attn_bias=attn_bias,
            window_attn_bias=attn_bias,
            block_list=block_list,
            context_lens_tensor=context_lens,
            seq_lens_tensor=torch.full_like(context_lens, self.num_query_per_req),
            slot_mapping=query_slots,
            block_size=self.block_size,
            # The explicit bias carries both causality and window alignment.
            causal=False,
        )
        with runner.profiler.record_event("internal", "dflash2_draft_model"):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=query_positions,
                inputs_embeds=None,
                attn_metadata=metadata,
            )
            hidden_states = hidden_states.reshape(batch_size, self.num_query_per_req, -1)[:, 1:]
        with runner.profiler.record_event("internal", "dflash2_candidate_selector"):
            selected = self._model_api().select_dflash2_candidates(
                hidden_states,
                input_ids[:, 0],
            )
            return selected[:real_batch_size]

    @torch.inference_mode()
    def propose_query_block(
        self,
        bonus_token_ids: Sequence[int],
        query_starts: Sequence[int],
        block_table: torch.Tensor,
        row_indices: Sequence[int],
        runner,
        padded_batch_size: int | None = None,
    ) -> torch.Tensor:
        real_batch_size = len(bonus_token_ids)
        if real_batch_size == 0:
            return torch.empty((0, self.num_speculative_tokens), dtype=torch.int32, device=self.device)
        if len(query_starts) != real_batch_size or len(row_indices) != real_batch_size:
            raise ValueError("bonus ids, query starts, and row indices must have the same length")
        batch_size = real_batch_size if padded_batch_size is None else int(padded_batch_size)
        if batch_size < real_batch_size:
            raise ValueError(f"padded_batch_size={batch_size} is smaller than real batch {real_batch_size}")

        padded_query_starts = list(query_starts) + [0] * (batch_size - real_batch_size)
        padded_row_indices = list(row_indices) + [-1] * (batch_size - real_batch_size)

        query_positions_cpu = [
            list(range(int(start),
                       int(start) + self.num_query_per_req)) for start in padded_query_starts
        ]
        query_slots = self.make_slot_mapping(block_table, padded_row_indices, query_positions_cpu, runner)
        metadata = self._build_attention_metadata(
            padded_query_starts,
            query_slots,
            block_table,
            padded_row_indices,
            runner,
        )

        input_ids = torch.full(
            (batch_size, self.num_query_per_req),
            self.mask_token_id,
            dtype=torch.int32,
            device="cpu",
        )
        input_ids[:real_batch_size, 0] = torch.tensor(bonus_token_ids, dtype=torch.int32)
        input_ids = async_h2d_copy(input_ids, device=self.device)
        positions = async_h2d_copy(torch.tensor(query_positions_cpu, dtype=torch.int64), device=self.device)

        with runner.profiler.record_event("internal", "dflash2_draft_model"):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=None,
                attn_metadata=metadata,
            )
            hidden_states = hidden_states.reshape(batch_size, self.num_query_per_req, -1)[:, 1:]
        with runner.profiler.record_event("internal", "dflash2_candidate_selector"):
            model_api = self._model_api()
            selected = model_api.select_dflash2_candidates(
                hidden_states,
                input_ids[:, 0],
            )
            return selected[:real_batch_size]


__all__ = ["HpuDFlash2Proposer"]
