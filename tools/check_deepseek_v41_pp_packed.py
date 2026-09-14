# SPDX-License-Identifier: Apache-2.0
"""Isolate PP communication from native TP replay without loading model weights."""

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").replace("{rank}", str(rank))
from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment  # noqa: E402

prepare_environment()

import torch  # noqa: E402
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import (  # noqa: E402
    destroy_distributed_environment, destroy_model_parallel, init_distributed_environment, initialize_model_parallel,
)
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime  # noqa: E402
from vllm_gaudi.ops.tp2_model_adapter import DEEPSEEK_V41_PP1  # noqa: E402
from vllm_gaudi.ops.tp2_prepared_plan import (  # noqa: E402
    collect_prepared_group_replays, prepared_group_stats, record_native_decoder_outputs,
    recapture_native_decoder_programs, replay_native_decoder, shutdown_prepared_group_plans,
)
from vllm_gaudi.v1.worker.deepseek_v41_runner import PPBuffers  # noqa: E402
from vllm_gaudi.models.deepseek_v41_program import PreparedStage  # noqa: E402


class CompletionHead(torch.nn.Module):
    sample_greedy = PreparedStage.sample_greedy
    forward = PreparedStage.sample_greedy_commit
    # The compiled completion path calls the same projection helper as the
    # production PreparedStage.  Keep the transport harness' tiny head
    # structurally compatible instead of falling back to an eager callback.
    _head_projection = PreparedStage._head_projection

    def __init__(self, tp_rank):
        super().__init__()
        from vllm.distributed import tensor_model_parallel_all_gather
        self.pp_rank, self.tp_rank = 1, tp_rank
        self.bf16_head = False
        self.all_gather = tensor_model_parallel_all_gather
        weights = torch.zeros(8, 5120, dtype=torch.float32)
        weights[:, 0] = torch.arange(8) + tp_rank * 8 - 4
        self.weights = SimpleNamespace(head=SimpleNamespace(weight=weights.to("hpu")))


def program(hidden, pre, positions):
    value = hidden[:, 0, :] + positions.to(torch.bfloat16).unsqueeze(-1)
    value = (value + torch.ops.vllm_gaudi.tp2_exchange_peer(value)) * 0.5
    value = (value + torch.ops.vllm_gaudi.tp2_exchange_peer(value)) * 0.5
    return value, pre + 1, value + 2


def position_program(hidden, pre, positions):
    value = hidden[:, 0, :] + 1
    value = (value + torch.ops.vllm_gaudi.tp2_exchange_peer(value)) * 0.5
    value = (value + torch.ops.vllm_gaudi.tp2_exchange_peer(value)) * 0.5
    return value, pre + positions.float().unsqueeze(-1), value + 2


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-receive", action="store_true")
    parser.add_argument("--local-input",
                        action="store_true",
                        help="Complete unrelated PP traffic, then use independent CPU-generated replay inputs")
    parser.add_argument("--restore-tp-context",
                        action="store_true",
                        help="Diagnostic only: insert an ordinary TP exchange after PP")
    parser.add_argument("--steps", type=int, default=34)
    parser.add_argument("--position-bank", action="store_true", help="Bind changing immutable position views directly")
    parser.add_argument("--device-commit",
                        action="store_true",
                        help="Compile real greedy head and completion after native TP")
    parser.add_argument("--native-pp-copy", action="store_true", help="Use bounded native DMA for C1 packets")
    parser.add_argument("--profile", action="store_true", help="Capture only steps after cold/capture/hot warmup")
    parser.add_argument("--verify-commit",
                        action="store_true",
                        help="Deprecated: ordinary int32 completion is always exercised")
    args = parser.parse_args()
    torch.hpu.set_device(rank)
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu
    bind_worker_cpu(rank)
    init_distributed_environment(world_size=4,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=rank,
                                 backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    initialize_tp2_fused_ar_norm_runtime()
    pp = PPBuffers(torch.device("hpu"), dspark=False)
    assert pp.packed is not None, "Packed C1 transport must be selected explicitly"
    assert pp.device_commit == args.device_commit, "Device completion selection must match the runtime profile"
    assert pp.packed.native_copy == args.native_pp_copy
    commit_head = None
    if args.device_commit and pp.group.is_last_rank:
        commit_head = torch.compile(CompletionHead(rank % 2), backend="hpu_backend", fullgraph=True, dynamic=False)
    owner = torch.nn.Identity()
    compiled = torch.compile(position_program if args.position_bank else program,
                             backend="hpu_backend",
                             fullgraph=True,
                             dynamic=False)
    from vllm_gaudi.ops.deepseek_v41_inputs import PositionBank
    bank = PositionBank(512, 6, "hpu") if args.position_bank else None
    # Use the production PP1 contract so the native graph captures all five
    # four-layer groups (and their 40 reductions), even in this transport
    # harness.  A one-group synthetic topology is intentionally rejected by
    # the explicit V4.1 dependency planner.
    topology = DEEPSEEK_V41_PP1
    fixed, records, profiler = None, [], None
    step_timings = []
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    # Raw transport covers all BF16 encodings across changing C1/C6 packets,
    # non-contiguous source order and FP32 negative-zero/nonfinite payloads.
    for generation, count in enumerate((6, 1, 1, 2, 1, 1, 1)):
        bits = (torch.arange(count * 20480, dtype=torch.int32) + generation * 20480).to(torch.int16)
        expected_hidden = bits.view(torch.bfloat16).reshape(count, 5120, 4).transpose(1, 2).contiguous()
        pbits = torch.tensor([0, -2147483648, 2139095040, 2143289410], dtype=torch.int32).repeat(count)
        expected_pre = pbits.view(torch.float32).reshape(count, 4)
        if pp.group.is_first_rank:
            # Transfer the non-contiguous source, not only a pre-flattened copy.
            hidden = bits.to("hpu").view(torch.bfloat16).reshape(count, 5120, 4).transpose(1, 2)
            if args.native_pp_copy and count == 1:
                hidden = hidden.contiguous()
            pre = expected_pre.to("hpu")
            pp.exchange({"hidden_states": hidden, "pre_mix": pre}, count, decode=count == 1)
        else:
            raw = pp.exchange(None, count, decode=count == 1)
            assert torch.equal(raw["hidden_states"].cpu().view(torch.int16), expected_hidden.view(torch.int16))
            assert torch.equal(raw["pre_mix"].cpu().view(torch.int32), expected_pre.view(torch.int32))
        consumed, tokens = pp.finish_single(count, 100 + generation)
        assert (consumed, tokens) == (count, [100 + generation])
    for step in range(args.steps):
        step_started = time.perf_counter()
        if args.profile and step == 4:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                record_shapes=True,
                with_stack=False)
            profiler.start()
            recapture_native_decoder_programs()
        value = step % 32
        if args.device_commit and step % 2:
            value = -value
        deferred = None
        position = (0, 6, 127, 128, 511, 0, 1, 506)[step % 8] if bank is not None else value
        print(f"RANK {rank} STEP {step} PP start", flush=True)
        if pp.group.is_first_rank:
            hidden = torch.full((1, 4, 5120), value + rank, dtype=torch.bfloat16, device="cpu").to("hpu")
            pre = torch.full((1, 4), float(value), dtype=torch.float32, device="cpu").to("hpu")
            pp.exchange({"hidden_states": hidden, "pre_mix": pre}, 1, decode=True)
        else:
            received = pp.exchange(None, 1, decode=True)
            if args.wait_receive or args.local_input:
                torch.hpu.synchronize()
            if args.local_input:
                received = dict(hidden_states=torch.full((1, 4, 5120),
                                                         value + rank % 2,
                                                         dtype=torch.bfloat16,
                                                         device="cpu").to("hpu"),
                                pre_mix=torch.full((1, 4), float(value), dtype=torch.float32, device="cpu").to("hpu"))
            positions = (bank.view(position, 1)
                         if bank is not None else torch.tensor([value], dtype=torch.int32, device="cpu").to("hpu"))
            live = received["hidden_states"], received["pre_mix"], positions
            if args.restore_tp_context:
                probe = torch.zeros(1, 5120, dtype=torch.bfloat16, device="hpu")
                torch.ops.vllm_gaudi.tp2_exchange_peer(probe)
                torch.hpu.synchronize()
            if fixed is None:
                fixed = tuple(value.clone() for value in live)
            roots = dict(hidden_states=live[0],
                         pre_mix=live[1],
                         positions=live[2],
                         residual=None,
                         metadata=SimpleNamespace(is_prompt=False, native_completion=None))
            result = replay_native_decoder(owner, **roots)
            if result is None:
                for destination, source in zip(fixed, live, strict=True):
                    destination.copy_(source)
                fixed_roots = dict(roots, hidden_states=fixed[0], pre_mix=fixed[1], positions=fixed[2])
                with collect_prepared_group_replays(owner=owner,
                                                    adapter=topology,
                                                    snapshot=lambda: SimpleNamespace(restore=lambda: None),
                                                    **fixed_roots):
                    result = compiled(*fixed)
                    record_native_decoder_outputs(*result)
            print(f"RANK {rank} STEP {step} native submitted " + json.dumps(prepared_group_stats()), flush=True)
            ticket = roots["metadata"].native_completion
            if ticket is not None and not args.device_commit:
                deadline = time.monotonic() + 15
                while not ticket.query():
                    if time.monotonic() >= deadline:
                        diagnostic = dict(rank=rank,
                                          step=step,
                                          status="native-completion-timeout",
                                          native=prepared_group_stats())
                        evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
                        (evidence / f"timeout-rank{rank}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
                        raise RuntimeError(f"Native PP diagnostic completion timed out: {diagnostic}")
                    time.sleep(0.01)
            expected = ((value + 1.5, value + position, value + 3.5) if bank is not None else
                        (2 * value + 0.5, value + 1, 2 * value + 2.5))
            if args.device_commit:
                commit_head(result[0], pp.commit)
                deferred = result, expected
            else:
                actual = [value.cpu() for value in result]
                for actual_value, scalar in zip(actual, expected, strict=True):
                    assert torch.equal(actual_value, torch.full_like(actual_value, scalar)), (rank, step, scalar)
            records.append({"step": step, "native": prepared_group_stats()})
            print(f"RANK {rank} STEP {step} outputs exact", flush=True)
        if args.device_commit:
            consumed, actual_output = pp.finish_single_device()
            first_output = value + 1.5 if bank is not None else 2 * value + 0.5
            assert (consumed, actual_output) == (1, [15 if first_output > 0 else 0])
            if deferred is not None:
                result, expected = deferred
                for actual_value, scalar in zip((item.cpu() for item in result), expected, strict=True):
                    assert torch.equal(actual_value, torch.full_like(actual_value, scalar)), (rank, step, scalar)
        else:
            consumed, actual_output = pp.finish_single(1, 100 + step % 32)
            assert (consumed, actual_output) == (1, [100 + step % 32])
        pp.drain()
        pp.group.barrier()
        step_timings.append((time.perf_counter() - step_started) * 1000.0)
    if profiler is not None:
        profiler.stop()
        profiler.export_chrome_trace(str(evidence / f"rank{rank}.trace.json.gz"))
        recapture_native_decoder_programs()
    from vllm_gaudi.distributed.tp2_fused_ar_norm import _verify_prepared_runtime
    _verify_prepared_runtime(Path(os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"]))
    (evidence / f"rank{rank}.json").write_text(
        json.dumps(dict(
            status="exact",
            diagnostic="Packed C1 PP / ordinary completion / native TP dependencies; no model speed qualification",
            packet_generations=pp.packed.generation,
            packet_completed=pp.packed.completed,
            direct_position_views=args.position_bank,
            device_completion=args.device_commit,
            native_pp_copy=args.native_pp_copy,
            wait_receive=args.wait_receive,
            local_input=args.local_input,
            profile=args.profile,
            verify_commit=args.verify_commit,
            runtime_fingerprints="Qualified Bridge ABI manifest rechecked after execution",
            restore_tp_context=args.restore_tp_context,
            records=records,
            step_timings_ms=step_timings,
            pp_sends=pp.sends,
            pp_receives=pp.receives,
            native=prepared_group_stats()),
                   indent=2) + "\n")
    print(f"RANK {rank} shutdown native plans", flush=True)
    shutdown_prepared_group_plans()
    print(f"RANK {rank} shutdown model parallel groups", flush=True)
    destroy_model_parallel()
    print(f"RANK {rank} shutdown distributed environment", flush=True)
    destroy_distributed_environment()
    print(f"RANK {rank} shutdown complete", flush=True)


if __name__ == "__main__":
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2))):
        main()
