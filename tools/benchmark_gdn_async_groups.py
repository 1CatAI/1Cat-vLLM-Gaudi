# SPDX-License-Identifier: Apache-2.0
"""Stateful TP2 group pipeline qualification, without a profiler.

The serial control uses the same new functional-state group as the asynchronous
candidate. It is not a rerun of the historical in-place model baseline.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import habana_frameworks.torch  # noqa: F401
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime, _resolve_runtime
from vllm_gaudi.models.qwen3_next import HpuQwen3DecoderLayerGroup, HpuQwen3AsyncStateGroup
from vllm_gaudi.ops.flashinfer_gaudi_adapter import maybe_run_gdn_fused_decode_step
from vllm_gaudi.ops.gdn_async_state import GDNStateDMAPipeline, GDNQueuedStateDMAPipeline, copy_states_native


class Attention(torch.nn.Module):

    def __init__(self, state, conv, tensors, index):
        super().__init__()
        self._hpu_active_ssm_state = state
        self._hpu_defer_ssm_writeback = False
        self._hpu_pending_ssm_state = None
        self.conv = conv
        self.tensors = tensors
        self.indices = torch.tensor([index], dtype=torch.int32, device="hpu")

    def prepare_decode_state_view(self, *args):
        pass

    def forward(self, packed):
        a, b, a_log, dt_bias, conv_weight, projection = self.tensors
        result = maybe_run_gdn_fused_decode_step(mixed_qkv=packed,
                                                 a=a,
                                                 b=b,
                                                 A_log=a_log,
                                                 dt_bias=dt_bias,
                                                 conv_state=self.conv,
                                                 conv_weight=conv_weight,
                                                 conv_bias=None,
                                                 ssm_state=self._hpu_active_ssm_state,
                                                 load_state_indices=self.indices,
                                                 direct_conv_state=True,
                                                 direct_gdn_state=True,
                                                 direct_state_group_count=3,
                                                 direct_state_group_offset=0,
                                                 scale=128**-0.5,
                                                 state_is_active_view=True,
                                                 defer_state_writeback=self._hpu_defer_ssm_writeback)
        if result is None:
            raise RuntimeError("Functional-state pipeline unexpectedly fell back")
        output, updated = result
        if self._hpu_defer_ssm_writeback:
            self._hpu_pending_ssm_state = updated
        return output.reshape(1, 3072) @ projection


class Layer(torch.nn.Module):

    def __init__(self, attention, weight):
        super().__init__()
        self.linear_attn = attention
        self.weight = weight

    def forward(self, *, positions, hidden_states, residual):
        partial = self.linear_attn(hidden_states)
        return torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm(partial, residual, self.weight, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-bank", action="store_true")
    parser.add_argument("--copy-backend", choices=("pytorch", "native", "queued"), default="pytorch")
    parser.add_argument("--layers-per-group", type=int, default=2)
    args = parser.parse_args()
    assert 1 <= args.layers_per_group <= 8
    layer_count = args.layers_per_group * 3
    rank = int(os.environ["LOCAL_RANK"])
    os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
    # Keep each worker on a separate available physical core for this probe.
    if os.environ.get("ASYNC_DMA_PIN_SMALL_CORES", "1") == "1":
        os.sched_setaffinity(0, {4, 60} if rank == 0 else {22, 78})
    config = VllmConfig()
    init_distributed_environment(2, rank, "env://", rank, backend="hccl")
    with set_current_vllm_config(config):
        initialize_model_parallel(2, 1)
    initialize_tp2_fused_ar_norm_runtime()
    bridge = _resolve_runtime()[0]
    rng = torch.Generator().manual_seed(2815)

    def rand(*shape, dtype=torch.bfloat16, amplitude=0.03):
        return (torch.randn(*shape, generator=rng) * amplitude).to(device="hpu", dtype=dtype)

    tensors = (rand(1, 24), rand(1, 24), rand(24, dtype=torch.float32) - 2, rand(24), rand(5120, 4),
               rand(3072, 5120, amplitude=0.01))
    norm_weight = torch.ones(5120, dtype=torch.bfloat16, device="hpu")
    bank_count = 2 if args.shared_bank else 4
    pool_rows = max(98, 2 + 32 * ((layer_count - 1) // bank_count))
    banks = [rand(pool_rows, 24, 128, 128, dtype=torch.float32, amplitude=0.01) for _ in range(bank_count)]
    banks1 = [tensor.clone() for tensor in banks]
    initial = [tensor.cpu() for tensor in banks]
    convolutions = [rand(1, 3, 5120) for _ in range(layer_count)]
    conv1 = [tensor.clone() for tensor in convolutions]
    copy_function = copy_states_native if args.copy_backend == "native" else None
    pipelines = (GDNStateDMAPipeline(asynchronous=False, copy_function=copy_function),
                 GDNStateDMAPipeline(asynchronous=True, copy_function=copy_function))
    if args.copy_backend == "queued":
        pipelines = (GDNQueuedStateDMAPipeline(asynchronous=False), GDNQueuedStateDMAPipeline(asynchronous=True))
    groups = [[], []]
    active_rows = {bank: [] for bank in range(bank_count)}
    for index in range(layer_count):
        active_rows[index % bank_count].append(1 + 32 * (index // bank_count))
    for arm, (pool, conv) in enumerate(((banks, convolutions), (banks1, conv1))):
        for group_id in range(3):
            layers = []
            for index in range(group_id * args.layers_per_group, (group_id + 1) * args.layers_per_group):
                start = 1 + 32 * (index // bank_count)
                state = pool[index % bank_count].narrow(0, start, 1)
                layers.append(Layer(Attention(state, conv[index], tensors, start), norm_weight))
            group = HpuQwen3DecoderLayerGroup(tuple(layers))
            compiled = torch.compile(group, backend="hpu_backend", fullgraph=True, dynamic=False)
            groups[arm].append(HpuQwen3AsyncStateGroup(compiled, group, group_id, pipelines[arm]))
    packed = rand(1, 5120)
    positions = torch.zeros(1, dtype=torch.int64, device="hpu")
    from types import SimpleNamespace

    metadata = SimpleNamespace(is_prompt=False, direct_gdn_state=True)

    def token(arm):
        hidden, residual = packed, torch.zeros_like(packed)
        for group in groups[arm]:
            hidden, residual = group(positions=positions, hidden_states=hidden, residual=residual)
        return hidden

    def compare():
        for index, (left, right) in enumerate(zip(banks, banks1, strict=True)):
            x, y = left.cpu(), right.cpu()
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            unused = [row for row in range(pool_rows) if row not in active_rows[index]]
            torch.testing.assert_close(y[unused], initial[index][unused], rtol=0, atol=0)
        for left, right in zip(convolutions, conv1, strict=True):
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)

    launch_deltas, dma_deltas = [], []
    with torch.inference_mode(), set_forward_context(metadata, config):
        first = groups[0][0]
        originals = [attention._hpu_active_ssm_state.cpu().clone() for attention in first._state_layers]
        _, _, fresh_states = first._compiled(positions=positions,
                                             hidden_states=packed,
                                             residual=torch.zeros_like(packed),
                                             return_state_updates=True)
        torch.hpu.synchronize()
        for attention, original, fresh in zip(first._state_layers, originals, fresh_states, strict=True):
            torch.testing.assert_close(attention._hpu_active_ssm_state.cpu(), original, rtol=0, atol=0)
            assert attention._hpu_active_ssm_state.data_ptr() != fresh.data_ptr()
        for left, right in zip(convolutions[:args.layers_per_group], conv1[:args.layers_per_group], strict=True):
            left.copy_(right)
        torch.hpu.synchronize()
        print("FUNCTIONAL_CACHE_UNCHANGED", rank, flush=True)
        for step in range(12):
            packed.copy_(rand(1, 5120, amplitude=(0.0, 1e-4, 0.1, 1.0)[step % 4]))
            before = bridge.collective_launch_count()
            dma_before = bridge.gdn_state_dma_counts() if args.copy_backend != "pytorch" else None
            left, right = token(0), token(1)
            # Extra allocations while the DMA stream can still be live.
            churn = [
                torch.empty((1, 24, 128, 128), dtype=torch.float32, device="hpu").fill_(step + 13) for _ in range(8)
            ]
            del churn
            torch.hpu.synchronize()
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
            compare()
            delta = bridge.collective_launch_count() - before
            assert delta == 2 * layer_count, delta
            launch_deltas.append(delta)
            if dma_before is not None:
                dma_delta = [new - old for new, old in zip(bridge.gdn_state_dma_counts(), dma_before, strict=True)]
                assert dma_delta == [6, 2 * layer_count, 2 * layer_count * 24 * 128 * 128 * 4], dma_delta
                dma_deltas.append(dma_delta)
            print("EXACT_PIPELINED_STEP", rank, step, flush=True)
        # Queue multiple recurrent tokens without a host/device drain between
        # them, forcing real next-consumer dependencies and allocator reuse.
        for _ in range(8):
            left, right = token(0), token(1)
            torch.empty((12, 24, 128, 128), device="hpu").fill_(7)
        torch.hpu.synchronize()
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
        compare()
        # An indexed/prefill-style pool reset must be ordered after writes.
        token(0)
        token(1)
        reset_index = torch.tensor([1], dtype=torch.int64, device="hpu")
        for pipeline, pool in zip(pipelines, (banks, banks1), strict=True):
            pipeline.wait_all()
            for tensor in pool:
                tensor.index_fill_(0, reset_index, 0)
        left, right = token(0), token(1)
        torch.hpu.synchronize()
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
        compare()

        device, drained = [[], []], [[], []]
        for arm in range(2):
            for _ in range(3):
                token(arm)
            pipelines[arm].wait_all()
            torch.hpu.synchronize()
        for wave in range(7):
            for arm in ((0, 1) if wave % 2 == 0 else (1, 0)):
                start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start.record()
                for _ in range(12):
                    token(arm)
                pipelines[arm].wait_all()
                end.record()
                torch.hpu.synchronize()
                device[arm].append(start.elapsed_time(end) / 12)
        for wave in range(15):
            for arm in ((0, 1) if wave % 2 == 0 else (1, 0)):
                begin = time.perf_counter()
                token(arm)
                pipelines[arm].wait_all()
                torch.hpu.synchronize()
                drained[arm].append((time.perf_counter() - begin) * 1000)
    stats = [
        dict(submitted_groups=p.submitted_groups,
             submitted_bytes=p.submitted_bytes,
             consumer_waits=p.consumer_waits,
             flush_waits=p.flush_waits,
             host_synchronizations=p.host_synchronizations) for p in pipelines
    ]
    result = dict(status="pass",
                  rank=rank,
                  copy_backend=args.copy_backend,
                  layers_per_group=args.layers_per_group,
                  pool_rows=pool_rows,
                  shared_bank=args.shared_bank,
                  active_rows=active_rows,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arms=["serial_functional_commit", "async_functional_commit"],
                  compute_stream=str(torch.hpu.current_stream()),
                  copy_stream=str(pipelines[1].copy_stream),
                  device_ms=device,
                  drained_ms=drained,
                  device_median_ms=list(map(statistics.median, device)),
                  drained_median_ms=list(map(statistics.median, drained)),
                  functional_cache_unchanged=True,
                  changing_steps_exact=12,
                  queued_steps_exact=8,
                  allocation_churn=True,
                  reset_exact=True,
                  collective_launch_deltas=launch_deltas,
                  native_dma_deltas=dma_deltas,
                  stats=stats,
                  limits=[
                      f"Three {args.layers_per_group}-layer GDN groups; no full-model or NIC/MME overlap claim.",
                      "Runtime, physical modules and CPU affinity are recorded in the case contract."
                  ])
    args.output.with_name(f"{args.output.stem}-rank{rank}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    for pipeline in pipelines:
        pipeline.synchronize()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
