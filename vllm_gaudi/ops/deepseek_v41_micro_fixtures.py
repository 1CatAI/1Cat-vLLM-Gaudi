# SPDX-License-Identifier: Apache-2.0
"""Export bounded actual C6 warmup tensors without touching scored requests."""
from pathlib import Path

import torch


_captured = {}


def export_request_fixture(program, hidden, auxiliary, cursor, state, directory, request):
    """Record three actual C6 request boundaries in a separate unscored run.

    This deliberately synchronizes. Never enable it for timing or trace
    qualification; startup warmup tensors cannot stand in for 16K requests.
    """
    rank = torch.distributed.get_rank()
    key = ("request", request.req_id, directory)
    index = _captured.get(key, 0)
    if index >= 3 or getattr(request, "decode_start", 0) < 16384:
        return
    parameters = request.sampling_params
    if (parameters.temperature, parameters.top_p, parameters.seed) != (1., .95, 42):
        raise ValueError("Production fixtures require the official sampled request")
    torch.hpu.synchronize()
    root = Path(directory) / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=True)
    from vllm_gaudi.ops.deepseek_v41_replay import stage_state_tensors
    from vllm_gaudi.ops.tp2_prepared_plan import _modules

    # Stream decoder state separately instead of retaining gigabytes of CPU
    # clones alongside the resident model. Keep its full physical page range:
    # allocator pages need not be the first 16K logical rows.
    state_ids = {id(value) for value in stage_state_tensors(program)}
    state_directory = root / f"c6-{index}-decoder-state"
    state_directory.mkdir()
    state_files = {}
    for name, value in program.named_buffers():
        if id(value) in state_ids:
            path = state_directory / (name + ".pt")
            torch.save(value.cpu(), path)
            state_files[name] = str(path.relative_to(root))
    ids, positions = cursor.ids.cpu().clone(), cursor.positions.cpu().clone()
    owners = [owner for owner in program.replay_owner.variants.values()
              if owner.fixed[3] is not None and owner.fixed[3].numel() == 6
              and torch.equal(owner.fixed[3].cpu(), ids) and torch.equal(owner.fixed[2].cpu(), positions)]
    groups = {}
    if len(owners) == 1:
        for module in tuple(_modules):
            for plan, plan_owner in zip(module.plans, module.plan_owners, strict=True):
                if plan_owner is None or plan_owner[0] != id(owners[0]):
                    continue
                outputs = plan.outputs()
                if len(outputs) >= 2 and outputs[0].shape == (6, 4, 5120) and outputs[1].shape == (6, 4):
                    groups[plan_owner[1]] = dict(residual=outputs[0].cpu().clone(), pre=outputs[1].cpu().clone())
    fixture = dict(source="actual 16K official sampled request; unscored fixture export",
                   request_context_qualified=True, request_id=request.req_id,
                   context_prefix_tokens=request.decode_start, rank=rank,
                   tensor_parallel_size=program.tensor_parallel_size,
                   hidden=hidden.cpu().clone(), auxiliary=auxiliary.cpu().clone(),
                   logits=program.draft._head_projection(hidden).cpu().clone(),
                   ids=ids, positions=positions, groups=groups, groups_qualified=len(owners) == 1,
                   decoder_state_files=state_files, search_length=program.search_length,
                   control=cursor.control.cpu().clone(), proposal=state.proposal.cpu().clone(),
                   sampling_parameters=state.parameters.cpu().clone(), sampling_seed=state.seed.cpu().clone(),
                   sampling_counter=state.counter.cpu().clone(), sampling_offsets=state.offsets.cpu().clone(),
                   draft_swa=tuple(layer.attention.swa.cpu().clone() for layer in program.draft.layers))
    torch.save(fixture, root / f"c6-{index}.pt")
    _captured[key] = index + 1


def export_warmup_fixture(program, hidden, ids, positions, directory, *, proposal=None):
    from vllm_gaudi.ops.tp2_prepared_plan import _modules

    rank = torch.distributed.get_rank()
    key = (id(program), directory)
    index = _captured.get(key, 0)
    if index >= 3:
        return
    if hidden.shape != (6, 5120):
        raise ValueError("Warm fixture requires actual C6 normalized target hidden")
    torch.hpu.synchronize()
    cpu_ids, cpu_positions = ids.cpu().clone(), positions.cpu().clone()
    # Warmed search buckets retain stale outputs. Only export the variant whose
    # actual input values match this round, never an arbitrary last dictionary entry.
    owners = [v for v in program.replay_owner.variants.values()
              if v.fixed[3] is not None and v.fixed[3].numel() == 6
              and torch.equal(v.fixed[3].cpu(), cpu_ids)
              and torch.equal(v.fixed[2].cpu(), cpu_positions)]
    owner_ids = {id(owners[0])} if len(owners) == 1 else set()
    groups = {}
    for module in tuple(_modules):
        for plan, owner in zip(module.plans, module.plan_owners, strict=True):
            if owner is None or owner[0] not in owner_ids:
                continue
            values = plan.outputs()
            if len(values) >= 2 and values[0].shape == (6, 4, 5120) and values[1].shape == (6, 4):
                groups[owner[1]] = dict(residual=values[0].cpu().clone(), pre=values[1].cpu().clone())
    logits = program.draft._head_projection(hidden).cpu()
    fixture = dict(source="actual checkpoint decoder during non-scored official sampled warmup",
                   rank=rank, tensor_parallel_size=program.tensor_parallel_size,
                   hidden=hidden.cpu().clone(), logits=logits,
                   ids=cpu_ids, positions=cpu_positions, groups=groups,
                   groups_qualified=len(owners) == 1,
                   proposal=None if proposal is None else proposal.cpu().clone())
    root = Path(directory) / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, root / f"c6-{index}.pt")
    _captured[key] = index + 1
