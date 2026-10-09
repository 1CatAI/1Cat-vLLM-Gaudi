# SPDX-License-Identifier: Apache-2.0
"""Gate data-dependent fallback with the production TP peer/HCCL transport.

Synthetic changing-coverage checks establish backend capability only. Supplying
three to five warmup fixtures additionally times the actual checkpoint head ->
p/q -> rejection consumer with same-process AB3. No serving gain is implied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radix", action="store_true")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--width", type=int, choices=(64, 128, 256), default=64)
    args = parser.parse_args()
    if bool(args.fixtures) != bool(args.prepared):
        parser.error("Real head-chain timing requires both --fixtures and --prepared")
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace(
        "{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(tensor_parallel_size=size, pipeline_parallel_size=1)
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    import torch.nn.functional as F
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet, sample_nucleus_packet
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        bounded_proposal_distribution, sample_bounded_or_full_sharded,
        sample_full_distribution, sample_speculative_prefix_sharded, speculative_sampling_draws)

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(rank=rank, size=size, capability_passed=False, production_micro_qualified=False,
                  default_enabled=False, cases=[], rounds=[])
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=size))
    bind_worker_cpu(rank)
    torch.hpu.set_device(rank)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    if args.radix:
        torch.ops.load_library(os.environ["DSV41_UNIQUE_OPERATOR_LIBRARY"])
    init_distributed_environment(world_size=size, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
        initialize_tp2_fused_ar_norm_runtime()
    bind_worker_helpers(rank)
    _, gather = stage_collectives(rank, True, size)
    local_vocab = 32320

    def gather_fp32(value):
        return gather(value.contiguous().view(torch.bfloat16), dim=-1).contiguous().view(torch.float32)

    def reference(local, settings):
        token, probability = sample_full_distribution(gather_fp32(local), settings)
        return token, probability[:, rank * local_vocab:(rank + 1) * local_vocab].contiguous()

    def candidate(local, settings):
        return sample_bounded_or_full_sharded(local, settings, tp_rank=rank, tp_size=size,
                                             all_gather=gather, radix=args.radix, width=args.width)

    def certificate(local, settings):
        packet = gather_fp32(local_nucleus_packet(local, settings, rank, args.width, radix=args.radix))
        _, covered = sample_nucleus_packet(packet, settings, tp_size=size, width=args.width)
        _, probability_covered = bounded_proposal_distribution(
            packet, settings, tp_rank=rank, tp_size=size, width=args.width, local_vocab=local_vocab)
        return (covered & probability_covered & (settings[:, :1] > 0)).all().reshape(1)

    compile_fn = lambda fn: torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            parent, proposed, certify = map(compile_fn, (reference, candidate, certificate))
            for count in (5, 6):
                local = torch.empty((count, local_vocab), dtype=torch.float32, device="hpu")
                settings_cpu = torch.tensor([[1., .95, .13, -1.]]).repeat(count, 1)
                settings = settings_cpu.to("hpu")
                for kind, uniform in (("covered", .13), ("fallback", .77), ("covered", .41)):
                    full = torch.randn(count, size * local_vocab, generator=torch.Generator().manual_seed(42)) * .1
                    if kind == "covered":
                        full[:, :8] += torch.arange(8).float() * .3 + 15
                    settings_cpu[:, 2] = uniform
                    settings.copy_(settings_cpu)
                    shard = full[:, rank * local_vocab:(rank + 1) * local_vocab].contiguous()
                    local.copy_(shard)
                    expected = tuple(v.cpu() for v in parent(local, settings))
                    actual = tuple(v.cpu() for v in proposed(local, settings))
                    device_covered = bool(certify(local, settings).cpu()[0])
                    packets = torch.cat([local_nucleus_packet(
                        full[:, r * local_vocab:(r + 1) * local_vocab], settings_cpu, r, args.width)
                        for r in range(size)], -1)
                    _, cpu_covered = sample_nucleus_packet(packets, settings_cpu, tp_size=size, width=args.width)
                    if device_covered != bool(cpu_covered.all()) or device_covered != (kind == "covered"):
                        raise AssertionError("Coverage changed or candidate silently always fell back")
                    if args.radix:
                        selected, _ = torch.ops.custom_op.custom_deepseek_v41_vocab_radix_topk_gaudi2(local, args.width)
                        torch.testing.assert_close(selected.cpu().sort(-1, descending=True)[0],
                                                   shard.topk(args.width, dim=-1)[0], rtol=0, atol=0)
                    if not torch.equal(expected[0], actual[0]):
                        raise AssertionError("Changing collective fallback changed sampled tokens")
                    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=2e-7)
                    exact = torch.equal(expected[1], actual[1])
                    if kind == "fallback" and not exact:
                        raise AssertionError("Full fallback probability is not byte exact")
                    report["cases"].append(dict(rows=count, kind=kind, covered=device_covered,
                                                tokens_exact=True, fallback_probability_exact=exact,
                                                probability_max_abs=float((actual[1] - expected[1]).abs().max())))
            report["capability_passed"] = True
            if args.fixtures:
                from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

                paths = sorted((args.fixtures / f"rank{rank}").glob("c6-*.pt"))
                if not 3 <= len(paths) <= 5 or args.samples < 3:
                    raise ValueError("Production micro needs 3-5 real fixture sets and >=3 device samples")
                shard_loader = PreparedV41Shard(args.prepared, 0, rank)
                weight = shard_loader.tensor("head.weight", "hpu")
                if weight.dtype != torch.bfloat16 or weight.shape != (local_vocab, 5120):
                    raise ValueError("Unexpected checkpoint head layout or precision")
                weight = weight.float()
                fixtures = []
                report["fixtures"] = []
                for path in paths:
                    data = torch.load(path, weights_only=True, map_location="cpu")
                    if not data.get("request_context_qualified"):
                        raise ValueError("Startup warmup cannot qualify the actual 16K sampled protocol")
                    if data["rank"] != rank or data["tensor_parallel_size"] != size or data.get("proposal") is None:
                        raise ValueError("Need rank-matched real hidden and actual proposal probabilities")
                    if data["hidden"].shape != (6, 5120) or data["proposal"].shape != (5, local_vocab):
                        raise ValueError("Unexpected Target/proposal production shapes")
                    _, acceptance, correction, controls = speculative_sampling_draws(
                        data["sampling_parameters"].clone(), data["sampling_seed"],
                        data["sampling_counter"].clone(), data["sampling_offsets"])
                    operands = (data["hidden"], data["proposal"], data["ids"][1:],
                                controls, acceptance, correction)
                    fixtures.append(tuple(value.to("hpu") for value in operands))
                    report["fixtures"].append(dict(name=path.name,
                                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

                def chain(hidden, proposal_q, ids, controls, acceptance, correction, *, bounded):
                    logits = F.linear(hidden.float(), weight)
                    sampled, probability = (candidate if bounded else reference)(logits, controls)
                    result = sample_speculative_prefix_sharded(
                        probability, proposal_q, ids, acceptance, correction, rank, size, gather)
                    return *result, sampled

                def parent_chain(*inputs):
                    return chain(*inputs, bounded=False)

                def candidate_chain(*inputs):
                    return chain(*inputs, bounded=True)

                arms = dict(parent=compile_fn(parent_chain), candidate=compile_fn(candidate_chain))
                for inputs in fixtures:
                    expected = tuple(v.cpu() for v in arms["parent"](*inputs))
                    actual = tuple(v.cpu() for v in arms["candidate"](*inputs))
                    if any(not torch.equal(a, b) for a, b in zip(expected, actual, strict=True)):
                        raise AssertionError("Real producer/consumer changed committed tokens/count/status")
                for fn in arms.values():
                    for inputs in fixtures:
                        fn(*inputs)
                torch.hpu.synchronize()
                for iteration in range(3):
                    medians = {}
                    for name, fn in arms.items():
                        device_times = []
                        for sample in range(args.samples):
                            # Every rank walks the same shapes/collectives in the same order.
                            begin, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                            begin.record()
                            fn(*fixtures[sample % len(fixtures)])
                            end.record()
                            end.synchronize()
                            device_times.append(begin.elapsed_time(end))
                        medians[name] = statistics.median(device_times)
                        report["rounds"].append(dict(round=iteration, arm=name, device_ms=device_times,
                                                     median_device_ms=medians[name]))
                    report.setdefault("paired_saved_ms", []).append(medians["parent"] - medians["candidate"])
                report["saved_ms_per_round"] = statistics.median(report["paired_saved_ms"])
                report["production_micro_qualified"] = (all(v > 0 for v in report["paired_saved_ms"])
                                                         and report["saved_ms_per_round"] >= .3)
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        (root / f"sharded-bounded-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
