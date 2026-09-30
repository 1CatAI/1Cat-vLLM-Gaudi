# SPDX-License-Identifier: Apache-2.0
"""Submit the two index-query gathers together without a packed intermediate."""
import torch

_RUNTIME = "_vllm_gaudi_tp4_index_gather_pair_runtime"
_library = torch.library.Library("vllm_gaudi", "FRAGMENT")
_library.define("tp4_index_gather_pair(Tensor query, Tensor gains) -> (Tensor, Tensor)")


def _outputs(query, gains):
    return (torch.empty((query.numel() * 4,), dtype=query.dtype, device=query.device),
            torch.empty((gains.numel() * 4,), dtype=gains.dtype, device=gains.device))


def _gather(query, gains):
    runtime = getattr(torch, _RUNTIME, None)
    if runtime is None:
        raise RuntimeError("Prepare the TP4 index-gather runtime before compilation")
    bridge, backend = runtime
    outputs = _outputs(query, gains)
    return bridge.tp4_index_gather_pair(backend, query, gains, *outputs)


_library.impl("tp4_index_gather_pair", _gather, dispatch_key="HPU")
_library._register_fake("tp4_index_gather_pair", _outputs)


def prepare_index_gather_pair():
    """Validate and retain the ordinary four-rank communication owner once."""
    import torch.distributed as dist
    from vllm.distributed import get_tp_group

    from vllm_gaudi.ops.deepseek_v41_completion import _control_bridge

    group = get_tp_group()
    if group.world_size != 4:
        raise ValueError("Paired index gather requires TP4")
    backend = group.device_group._get_backend(torch.device("hpu"))
    existing = getattr(torch, _RUNTIME, None)
    if existing is not None:
        if existing[1] is not backend:
            raise RuntimeError("Paired index gather communicator changed without retirement")
        return existing[0]
    bridge = _control_bridge()
    if getattr(bridge, "tp4_index_gather_pair_version", 0) != 1:
        raise RuntimeError("Build the bridge with paired index-gather ownership support")
    probe = torch.ones(128, dtype=torch.bfloat16, device="hpu")
    dist.all_reduce(probe, group=group.device_group)
    if not torch.equal(probe.cpu(), torch.full((128,), 4, dtype=torch.bfloat16)):
        raise RuntimeError("Paired index-gather communicator initialization failed")
    setattr(torch, _RUNTIME, (bridge, backend))
    return bridge


def gather_index_query_pair(query, gains):
    """Restore the same batch/head order consumed by the existing score kernel."""
    if (query.ndim != 3 or query.shape[1:] != (8, 128) or gains.shape != query.shape[:2]
            or not 1 <= query.shape[0] <= 6 or query.dtype != torch.bfloat16 or gains.dtype != torch.bfloat16):
        raise ValueError("Paired index query/gains require TP4 C1..C6 BF16 head shards")
    batch = query.shape[0]
    query_rows, gain_rows = torch.ops.vllm_gaudi.tp4_index_gather_pair(query.contiguous(), gains.contiguous())
    return (query_rows.reshape(4, batch, 8, 128).permute(1, 0, 2, 3).reshape(batch, 32, 128),
            gain_rows.reshape(4, batch, 8).permute(1, 0, 2).reshape(batch, 32))
