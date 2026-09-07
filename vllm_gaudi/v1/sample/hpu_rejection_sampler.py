# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from vllm.v1.sample import rejection_sampler
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def _dflash2_greedy_fastpath_supported(sampling_metadata: SamplingMetadata, ) -> bool:
    """Return whether argmax can bypass the general sampler losslessly."""
    if not sampling_metadata.all_greedy:
        return False
    if sampling_metadata.max_num_logprobs is not None or sampling_metadata.logprob_token_ids:
        return False
    if not sampling_metadata.no_penalties:
        return False
    if sampling_metadata.allowed_token_ids_mask is not None or sampling_metadata.bad_words_token_ids:
        return False

    # Spec decode currently installs only MinTokens as an argmax-changing
    # processor. It is a true no-op once its active request map is empty.
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        if not isinstance(processor, MinTokensLogitsProcessor) or processor.min_toks:
            return False

    holder = sampling_metadata.thinking_budget_state_holder
    return holder is None or not holder.has_tracked_requests()


def dflash2_greedy_rejection_sample(
    logits: torch.Tensor,
    metadata: SpecDecodeMetadata,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor | None:
    """Fuse DFlash2's two greedy sampler passes into one vocabulary argmax.

    The general rejection sampler separately gathers/casts/samples bonus and
    target rows. For the common unconstrained greedy request, casting cannot
    change the argmax, so one argmax over the already-packed verification
    logits is exactly equivalent and avoids materializing FP32 vocabulary
    tensors twice.
    """
    if not _dflash2_greedy_fastpath_supported(sampling_metadata):
        return None

    return dflash2_greedy_rejection_sample_packed(
        logits,
        metadata.draft_token_ids,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.cu_num_draft_tokens,
    )


def dflash2_greedy_rejection_sample_packed(
    logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Tensor-only DFlash2 greedy sampler suitable for regional compile."""
    all_token_ids = logits.argmax(dim=-1).to(torch.int32)
    target_token_ids = torch.index_select(all_token_ids, 0, target_logits_indices)
    bonus_token_ids = torch.index_select(all_token_ids, 0, bonus_logits_indices).view(-1, 1)
    return _rejection_sample_pytorch(
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        bonus_logits_indices.shape[0],
        cu_num_draft_tokens,
    )


def rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Performs vectorized rejection sampling on a batch of token sequences.

    This function compares draft tokens to target tokens and accepts them up to 
    the first mismatch. If an entire sequence of draft tokens is accepted, a 
    bonus token is appended. This version handles variable numbers of draft 
    tokens per sequence.

    The current HPU implementation of spec decode will flatten the num_draft_tokens
    to 1. And so the shape size of padded_draft_token_ids will be
    the [real batch size * num_draft_tokens, 1].

    Args:
        padded_draft_token_ids (torch.Tensor): A 2D tensor of draft tokens.
            Shape: (num_seqs * max_draft_tokens, 1)
        padded_target_token_ids (torch.Tensor): A 1D tensor of target tokens
            predicted by the main model.
            Shape: (num_seqs * max_draft_tokens)
        bonus_token_ids (torch.Tensor): A single bonus token for each sequence,
            to be used if all draft tokens are accepted.
            Shape: (num_seqs, 1)
        num_draft_tokens: list[int]: List of number draft tokens for each sequence.
            Shape: (num_seqs)
        cu_num_draft_tokens (torch.Tensor): The cumulative sum of the number of
            draft tokens for each request. Used to determine actual sequence 
            lengths. Shape: (num_seqs,)

    Returns:
        torch.Tensor: The resulting tensor of accepted tokens.
            Shape: (num_seqs, max_draft_tokens + 1)
    """
    return _rejection_sample_pytorch(
        padded_draft_token_ids,
        padded_target_token_ids,
        bonus_token_ids,
        len(num_draft_tokens),
        cu_num_draft_tokens,
    )


def _rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    num_seqs: int,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    # Keep the prefix comparison and output packing on the input device.  The
    # caller already performs one final D2H copy when parsing engine output;
    # copying all draft/target rows here adds an extra synchronization to every
    # speculative step and is especially expensive for short decode batches.
    padded_draft_token_ids = padded_draft_token_ids.to(torch.int32)
    padded_target_token_ids = padded_target_token_ids.to(torch.int32)
    bonus_token_ids = bonus_token_ids.to(torch.int32)
    padded_draft_token_ids = padded_draft_token_ids.view(num_seqs, -1)
    max_draft_tokens = padded_draft_token_ids.shape[-1]
    padded_target_token_ids = padded_target_token_ids.view(num_seqs, -1)
    bonus_token_ids = bonus_token_ids.view(num_seqs, -1)
    device = padded_draft_token_ids.device

    # Calculate the number of draft tokens for each request without moving the
    # cumulative lengths to the host.
    cu_num_draft_tokens = cu_num_draft_tokens.to(device=device)
    start_indices = torch.cat((torch.zeros(1, device=device,
                                           dtype=cu_num_draft_tokens.dtype), cu_num_draft_tokens[:-1]))
    num_draft_tokens_per_seq = cu_num_draft_tokens - start_indices

    # Find the first mismatch while ignoring padded draft positions.
    pos = torch.arange(max_draft_tokens, device=device, dtype=num_draft_tokens_per_seq.dtype)
    valid_token_mask = pos < num_draft_tokens_per_seq.unsqueeze(-1)
    mismatches = (padded_draft_token_ids != padded_target_token_ids) & valid_token_mask
    any_mismatch = mismatches.any(dim=1)
    first_mismatch_idx = torch.argmax(mismatches.int(), dim=1)

    prefix_len = torch.where(any_mismatch, first_mismatch_idx + 1, num_draft_tokens_per_seq)
    prefix_mask = (pos < prefix_len.unsqueeze(-1)) & valid_token_mask
    placeholder = torch.full_like(padded_target_token_ids, PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    output_prefix = torch.where(prefix_mask, padded_target_token_ids, placeholder)

    output_tokens = torch.cat(
        (output_prefix, placeholder[:, :1]),
        dim=1,
    )
    output_pos = torch.arange(max_draft_tokens + 1, device=device, dtype=num_draft_tokens_per_seq.dtype)
    bonus_mask = (~any_mismatch).unsqueeze(-1) & (output_pos.unsqueeze(0) == num_draft_tokens_per_seq.unsqueeze(-1))
    return torch.where(bonus_mask, bonus_token_ids[:, :1], output_tokens)


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    # NOTE: HPU spec decode only supports greedy sampling, so the
    # `use_fp64_gumbel` knob (used by the random/gumbel recovery path upstream)
    # is accepted for signature parity but intentionally unused here.
    assert sampling_metadata.all_greedy, "Only greedy sampling is supported."

    # Rejection sampling for greedy sampling requests.

    target_argmax = target_probs.argmax(dim=-1)
    output_token_ids = rejection_sample_pytorch(draft_token_ids, target_argmax, bonus_token_ids, num_draft_tokens,
                                                cu_num_draft_tokens)
    return output_token_ids


rejection_sampler.rejection_sample = rejection_sample
