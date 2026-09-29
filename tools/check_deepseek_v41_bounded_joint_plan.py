# SPDX-License-Identifier: Apache-2.0
"""Native optional recipe publication with actual TP and mandatory consumers.

This is a runtime ownership/dependency gate, not a model performance result.
Eight mandatory exchanges and eight optional MME recipes share fixed buffers.
Every consumer masks stale optional outputs using the same input tile bound.
"""
import argparse
import json
import os
from pathlib import Path

rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
if os.environ.get("DSV41_MICRO_RANK_CPUS"):
    os.sched_setaffinity(0, json.loads(os.environ["DSV41_MICRO_RANK_CPUS"])[rank])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--whole-scorer", action="store_true")
    args = parser.parse_args()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime, _resolve_runtime
    from deepseek_v41_micro_replay import RecipeRecorder

    torch.set_num_threads(1)
    torch.manual_seed(151 + rank)
    torch.hpu.set_device(rank)
    config = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2)))
    config.__enter__()
    init_distributed_environment(world_size=2,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=rank,
                                 backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2)
    initialize_tp2_fused_ar_norm_runtime()
    bridge, backend, _ = _resolve_runtime()
    output = args.output.with_name(f"rank{rank}-" + args.output.name)
    directory = output.parent / f"rank{rank}"
    directory.mkdir(exist_ok=True)
    recorder = RecipeRecorder(directory, backend=backend)
    compiled = lambda fn: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    producer = compiled(lambda x: (x.float() * .125).bfloat16())
    tile = compiled(lambda x, w: torch.mm(x, w))

    def merge(partial, peer, bound, s0, s1, s2, s3, s4, s5, s6, s7, whole):
        values = torch.stack((s0, s1, s2, s3, s4, s5, s6, s7), 0).float()
        active = torch.arange(8, device=bound.device).reshape(8, 1, 1) < bound
        value = torch.where(active, values, 0).sum() * .0001
        if args.whole_scorer:
            value = torch.where(bound.reshape(()) > 4, whole.float().sum() * .0008, value)
        return (partial.float() + peer.float() + value).bfloat16()

    merger = compiled(merge)
    add = compiled(lambda x, peer: (x.float() + peer.float()).mul(.5).bfloat16())
    exchange = torch.ops.vllm_gaudi.tp2_exchange_peer.default
    x = torch.randn(1, 5120, device="hpu", dtype=torch.bfloat16)
    weight = torch.randn(5120, 128, device="hpu", dtype=torch.bfloat16) * .001
    bound_tensor = torch.tensor([8], dtype=torch.int32, device="hpu")
    # Prepare each recipe before recording. Cold discovery is not replay time.
    partial = producer(x)
    scores = [tile(x, weight) for _ in range(8)]
    merger(partial, partial, bound_tensor, *scores, scores[0])
    add(partial, partial)
    torch.hpu.synchronize()
    plan = bridge.PreparedGroupPlan()
    slots, external = {}, []

    def slot(value, is_input):
        if isinstance(value, torch.Tensor):
            key = (value.data_ptr(), tuple(value.shape), value.stride(), value.dtype)
        else:
            key = (type(value), value)
        if key not in slots:
            slots[key] = plan.add_slot(value, is_input)
            if is_input:
                external.append(value)
        return slots[key]

    def compute(fn, arguments, minimum=0):
        recorder.calls = []
        result = fn(*arguments)
        torch.hpu.synchronize()
        calls, recorder.calls = recorder.calls, None
        if not calls:
            raise RuntimeError("Compiled recipe recorder missed a compute node")
        for recipe, inputs, outputs in calls:
            plan.add_compute(recipe, [slot(v, True) for v in inputs], [slot(v, False) for v in outputs])
            if minimum:
                plan.mark_last_optional_tile(minimum)
        return result

    def peer(value):
        other = exchange(value)
        torch.hpu.synchronize()
        plan.add_peer_exchange(slot(value, False), slot(other, False))
        return other

    partial = compute(producer, (x, ))
    received = peer(partial)
    scores = [compute(tile, (x, weight), i + 1) for i in range(8)]
    whole = compute(tile, (x, weight), 9) if args.whole_scorer else scores[0]
    value = compute(merger, (partial, received, bound_tensor, *scores, whole))
    for _ in range(7):
        value = compute(add, (value, peer(value)))
    plan.prepare(backend, [slot(value, False)])
    graph = bridge.NativeDecodeGraph()
    graph.configure_topology(1, 8, False)
    graph.capture([plan], [external])
    graph.instantiate()
    graph.bind_dynamic_inputs([x, bound_tensor])
    report = {
        "scope": __doc__,
        "cases": [],
        "passed": False,
        "segments": graph.segment_count(),
        "collectives": graph.collective_count(),
        "workspace_bytes": graph.workspace_bytes()
    }
    save = lambda: output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    for iteration, bound in enumerate((8, 1, 0, 4, 5, 2, 0, 7, 1, 8)):
        torch.hpu.synchronize()
        x.copy_(torch.randn_like(x) * (iteration + 1) * .1)
        bound_tensor.fill_(bound)
        expected = producer(x)
        result = merger(expected, exchange(expected), bound_tensor, *(tile(x, weight) for _ in range(9)))
        for _ in range(7):
            result = add(result, exchange(result))
        expected = result.cpu()
        # Poison skipped outputs to prove that the mandatory consumer masks
        # them, and that skipped matrices really do not publish fresh values.
        skipped = list(range(8)) if args.whole_scorer and bound > 4 else list(range(bound, 8))
        for i in skipped:
            scores[i].fill_(float("nan"))
        if args.whole_scorer and bound <= 4:
            whole.fill_(float("nan"))
        torch.hpu.synchronize()
        ticket = graph.replay_bounded_fixed_with_completion(bound)
        ticket.synchronize()
        actual = value.cpu()
        stale = [bool(torch.isnan(scores[i]).all().item()) for i in skipped]
        if args.whole_scorer and bound <= 4:
            stale.append(bool(torch.isnan(whole).all().item()))
        case = {
            "iteration": iteration,
            "bound": bound,
            "exact": torch.equal(expected, actual),
            "skipped_outputs_remained_poisoned": stale,
            "joint_info": list(graph.joint_info())
        }
        report["cases"].append(case)
        save()
        if not case["exact"] or not all(stale):
            torch.save({"expected": expected, "actual": actual}, output.with_suffix(".failure.pt"))
            raise RuntimeError(f"Bounded joint plan failed: {case}")
    graph.reset_slots()
    graph.close()
    plan.invalidate()
    report["passed"] = True
    report["retirement"] = list(graph.retirement_info())
    save()
    # The recorder's registered compiler observer survives this function.
    # Drop its communicator reference before destroying process groups.
    recorder.backend = None
    ticket = graph = plan = None
    del backend
    import gc
    gc.collect()
    torch.hpu.synchronize()
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    destroy_model_parallel()
    destroy_distributed_environment()
    config.__exit__(None, None, None)


if __name__ == "__main__":
    main()
