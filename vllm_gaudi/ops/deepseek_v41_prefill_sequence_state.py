# SPDX-License-Identifier: Apache-2.0
"""Token ownership for full TP4 prompt residuals and Engram computation."""
import torch.distributed as dist


def can_sequence_prefill_state(stage, tokens):
    from vllm_gaudi import envs
    return (stage.tensor_parallel_size == 4 and not stage.dspark and tokens == 16384
            and envs.VLLM_HPU_DSV41_PREFILL_REGIONS and envs.VLLM_HPU_DSV41_PREFILL_MHC_INPUT
            and envs.VLLM_HPU_DSV41_PREFILL_MHC_POST)


def token_owner(total_tokens, rank):
    if not isinstance(total_tokens, int) or total_tokens < 4 or total_tokens % 4 or not 0 <= rank < 4:
        raise ValueError("TP4 prompt ownership requires four equal nonempty token intervals")
    rows = total_tokens // 4
    return slice(rank * rows, (rank + 1) * rows)


def gather_tokens(local, *, group):
    if dist.get_world_size(group) != 4 or local.shape[0] < 1 or not local.is_contiguous():
        raise ValueError("TP4 prompt gather requires a contiguous nonempty token shard")
    full = local.new_empty((local.shape[0] * 4, *local.shape[1:]))
    dist.all_gather_into_tensor(full, local, group=group)
    return full


def reduce_owned_tokens(partial, reduce):
    """Preserve the qualified AllReduce arithmetic before taking owned rows."""
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    if group.world_size != 4:
        raise ValueError("Token-owned prompt reduction requires TP4")
    return reduce(partial)[token_owner(partial.shape[0], group.rank_in_group)]


def exchange_engram_tokens(packet, *, group):
    """Transpose distributed Engram head ownership into token ownership."""
    import torch
    if packet.ndim != 3 or not packet.is_contiguous() or dist.get_world_size(group) != 4:
        raise ValueError("Engram exchange requires contiguous TP4 [tokens,heads,width]")
    token_owner(packet.shape[0], dist.get_rank(group))
    rows = packet.shape[0] // 4
    if packet.dtype == torch.uint8:
        if packet.shape[-1] % 4:
            raise ValueError("Packed Engram rows must support an exact I32 wire view")
        wire = packet.view(torch.int32)
    elif packet.dtype == torch.bfloat16:
        wire = packet
    else:
        raise ValueError("Engram exchange accepts BF16 or packed U8 rows")
    received = torch.empty_like(wire)
    dist.all_to_all_single(received, wire, group=group)
    received = received.view(packet.dtype)
    return received.reshape(4, rows, *packet.shape[1:]).permute(1, 0, 2, 3).reshape(rows, packet.shape[1] * 4,
                                                                                    packet.shape[2])


def replicate_owned_tail(local, retained, *, group):
    """The final token owner supplies the full decoder halo to every TP rank."""
    if dist.get_world_size(group) != 4 or not 1 <= retained <= local.shape[0]:
        raise ValueError("Replicated decoder tail must fit the final TP4 token shard")
    result = (local[-retained:].contiguous() if dist.get_rank(group) == 3 else local.new_empty(
        (retained, *local.shape[1:])))
    dist.broadcast(result, src=dist.get_global_rank(group, 3), group=group)
    return result


def sequence_hc_input(residual,
                      previous,
                      fn,
                      scale,
                      base,
                      norm,
                      eps,
                      hc_eps,
                      iterations,
                      packed_fn,
                      *,
                      group,
                      gather=True):
    """Keep the full-prompt arithmetic policy while computing only owned rows."""
    from vllm_gaudi.ops.deepseek_v41_math import prefill_hc_input
    _, pre, post, comb, local = prefill_hc_input(residual,
                                                 previous,
                                                 fn,
                                                 scale,
                                                 base,
                                                 norm,
                                                 eps,
                                                 hc_eps,
                                                 iterations,
                                                 packed_fn,
                                                 logical_tokens=residual.shape[0] * 4)
    return pre, post, comb, gather_tokens(local.contiguous(), group=group) if gather else local


def retire_prefill_layer(boundary):
    """Finish each full prompt layer before reusing the routed workspace."""
    if boundary is None:
        return
    boundary.record()
    boundary.synchronize()
