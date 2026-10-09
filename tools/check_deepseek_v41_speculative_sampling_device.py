# SPDX-License-Identifier: Apache-2.0
"""Check sampled C6 rejection and exact peer transport with changing draws.

This exercises probabilities and control, not the draft/Target model. It is
not an acceptance-rate or serving-performance qualification.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", action="store_true")
    parser.add_argument("--full-proposal", action="store_true")
    args = parser.parse_args()
    rank, size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace(
        "{rank}", str(rank))
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment(tensor_parallel_size=size, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        filtered_distribution,
        sample_speculative_prefix,
        sample_speculative_prefix_sharded,
        sample_proposal_sharded,
        bounded_proposal_distribution,
        full_sampling_distribution_sharded,
    )
    from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet, sample_nucleus_packet

    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=size))
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    init_distributed_environment(world_size=size, rank=rank, distributed_init_method="env://",
                                 local_rank=rank, backend="hccl")
    with set_current_vllm_config(config):
        initialize_model_parallel(tensor_model_parallel_size=size, pipeline_model_parallel_size=1)
        initialize_tp2_fused_ar_norm_runtime()
    bind_worker_helpers(rank)
    _, gather = stage_collectives(rank, True, size)

    def control(p, q, ids, draws, correction):
        return sample_speculative_prefix_sharded(p, q, ids, draws, correction, rank, size, gather)

    compiled = torch.compile(control, backend="hpu_backend", fullgraph=True, dynamic=False)
    report = dict(status="running", cases=[], serving_qualified=False, acceptance_measured=False)
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            torch.manual_seed(42)
            vocab = size * 32320
            logits = torch.randn(6, vocab) * 4
            if args.proposal or args.full_proposal:
                def proposal(local, controls):
                    if args.full_proposal:
                        token, q = full_sampling_distribution_sharded(
                            local, controls, tp_rank=rank, all_gather=gather)
                        return token, q, torch.ones_like(token, dtype=torch.bool)
                    return sample_proposal_sharded(local, controls, tp_rank=rank, tp_size=size,
                                                   width=512, all_gather=gather)

                compiled_proposal = torch.compile(proposal, backend="hpu_backend", fullgraph=True, dynamic=False)
                if args.full_proposal:
                    def full_reference(full, controls):
                        from vllm_gaudi.ops.deepseek_v41_sampling import sample_probabilities

                        p = filtered_distribution(full, controls[:, 0], controls[:, 1], controls[:, 3])
                        start = rank * 32320
                        token = sample_probabilities(full, controls, filtered=True).reshape(-1)
                        return token, p[:, start:start + 32320]

                    compiled_reference = torch.compile(full_reference, backend="hpu_backend", fullgraph=True,
                                                       dynamic=False)
                    full_device = logits[:5].contiguous().to("hpu")
                controls = torch.tensor([[1., .95, .13, -1.]]).repeat(5, 1)
                local = logits[:5, rank * 32320:(rank + 1) * 32320].contiguous().to("hpu")
                device_controls = controls.to("hpu")
                for uniform in (.13, .77):
                    controls[:, 2] = uniform
                    device_controls.copy_(controls)
                    packet = torch.cat([local_nucleus_packet(logits[:5, r * 32320:(r + 1) * 32320],
                                                            controls, r, 512) for r in range(size)], -1)
                    token, covered = sample_nucleus_packet(packet, controls, tp_size=size, width=512)
                    probability, _ = bounded_proposal_distribution(
                        packet, controls, tp_rank=rank, tp_size=size, width=512, local_vocab=32320)
                    if args.full_proposal:
                        from vllm_gaudi.ops.deepseek_v41_sampling import sample_probabilities

                        token = sample_probabilities(logits[:5], controls, filtered=True)
                        covered = torch.ones(5, 1, dtype=torch.bool)
                        probability = filtered_distribution(
                            logits[:5], controls[:, 0], controls[:, 1], controls[:, 3])
                        probability = probability[:, rank * 32320:(rank + 1) * 32320]
                    actual = tuple(t.cpu() for t in compiled_proposal(local, device_controls))
                    if not torch.equal(actual[0], token.reshape(-1)):
                        raise AssertionError("Actual proposal token differs from its CPU packet sampler")
                    if not torch.equal(actual[2], covered.reshape(-1)):
                        raise AssertionError("Proposal coverage differs")
                    if args.full_proposal:
                        native_reference = tuple(t.cpu() for t in compiled_reference(full_device, device_controls))
                        if not torch.equal(actual[0], native_reference[0]):
                            raise AssertionError("Full fallback differs from ordinary HPU sampling")
                        torch.testing.assert_close(actual[1], native_reference[1], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(actual[1], probability, rtol=2e-5, atol=2e-7)
                    support_difference = int(((actual[1] > 0) != (probability > 0)).sum())
                    report["cases"].append(dict(uniform=uniform, tokens=actual[0].tolist(),
                                                coverage=actual[2].tolist(),
                                                q_max_abs=float((actual[1] - probability).abs().max()),
                                                cpu_support_differences=support_difference,
                                                ordinary_hpu_exact=args.full_proposal))
                report["status"] = ("passed_full_proposal_fallback" if args.full_proposal
                                    else "passed_actual_proposal_transport")
                return
            # Markov-dependent proposal logits differ from the target rows.
            q_logits = logits[:5] + torch.randn(5, vocab) * .5
            p = filtered_distribution(logits, torch.ones(6), torch.full((6,), .95), torch.zeros(6))
            q = filtered_distribution(q_logits, torch.ones(5), torch.full((5,), .95), torch.zeros(5))
            ids = q.argmax(-1)
            local = slice(rank * 32320, (rank + 1) * 32320)
            operands = [p[:, local].contiguous().to("hpu"), q[:, local].contiguous().to("hpu"),
                        ids.to("hpu"), torch.zeros(5, device="hpu"), torch.tensor(.13, device="hpu")]
            for draws, correction in ((torch.zeros(5), .13), (torch.full((5,), .999), .77),
                                      (torch.tensor([0., 0., .999, 0., 0.]), .41)):
                operands[3].copy_(draws)
                operands[4].fill_(correction)
                reference = sample_speculative_prefix(p, q, ids, draws, torch.tensor(correction))
                actual = tuple(t.cpu() for t in compiled(*operands))
                if any(not torch.equal(a, b) for a, b in zip(actual, reference, strict=True)):
                    raise AssertionError(dict(expected=[t.tolist() for t in reference],
                                              actual=[t.tolist() for t in actual]))
                report["cases"].append(dict(committed=int(actual[1]), tokens=actual[0].tolist(), exact=True))
            report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (root / f"sample-control-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        torch.hpu.synchronize()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
