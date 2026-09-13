# SPDX-License-Identifier: Apache-2.0
"""Qualify active cache views through HPU compilation without a profiler.

This is a three-layer recurrent chain with a downstream BF16 projection,
not an end-to-end model or a measurement of NIC/compute overlap.
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

from vllm_gaudi.ops.flashinfer_gaudi_adapter import maybe_run_gdn_fused_decode_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp2", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    bridge = None
    if args.tp2:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime, _resolve_runtime

        os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
        init_distributed_environment(2, rank, "env://", rank, backend="hccl")
        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel(2, 1)
        initialize_tp2_fused_ar_norm_runtime()
        bridge = _resolve_runtime()[0]
    generator = torch.Generator().manual_seed(58913)

    def rand(*shape, dtype=torch.bfloat16, scale=0.05):
        return (torch.randn(*shape, generator=generator) * scale).to(dtype=dtype, device="hpu")

    packed, a, b = rand(1, 5120), rand(1, 24), rand(1, 24)
    a_log, dt_bias, conv_weight = rand(24, dtype=torch.float32) - 2, rand(24), rand(5120, 4)
    projection = rand(3072, 5120, scale=0.01)
    norm_weight = torch.ones(5120, dtype=torch.bfloat16, device="hpu")
    pool0 = rand(98, 24, 128, 128, dtype=torch.float32, scale=0.01)
    pool1 = pool0.clone()
    initial = pool0.cpu()
    conv0 = tuple(rand(1, 3, 5120) for _ in range(3))
    conv1 = tuple(state.clone() for state in conv0)
    views = tuple(pool1.narrow(0, start, 1) for start in (1, 33, 65))
    indices = tuple(torch.tensor([start], dtype=torch.int32, device="hpu") for start in (1, 33, 65))

    def chain(p, gates_a, gates_b, conv, state, active):
        residual = torch.zeros_like(p)
        for offset in range(3):
            result = maybe_run_gdn_fused_decode_step(mixed_qkv=p,
                                                     a=gates_a,
                                                     b=gates_b,
                                                     A_log=a_log,
                                                     dt_bias=dt_bias,
                                                     conv_state=conv[offset],
                                                     conv_weight=conv_weight,
                                                     conv_bias=None,
                                                     ssm_state=state[offset] if active else state,
                                                     load_state_indices=indices[offset],
                                                     direct_conv_state=True,
                                                     direct_gdn_state=True,
                                                     direct_state_group_count=3,
                                                     direct_state_group_offset=offset,
                                                     state_is_active_view=active,
                                                     scale=128**-0.5)
            if result is None:
                raise RuntimeError("Active-state microbenchmark unexpectedly fell back")
            p = result[0].reshape(1, 3072) @ projection
            if args.tp2:
                p, residual = torch.ops.vllm_gaudi.tp2_allreduce_residual_rms_norm(p, residual, norm_weight, 1e-6)
        return p

    fn = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
    calls = ((packed, a, b, conv0, pool0, False), (packed, a, b, conv1, views, True))
    launches = []
    for step in range(12):
        packed.copy_(rand(1, 5120, scale=(0.0, 1e-4, 0.1, 1.0)[step % 4]))
        a.copy_(rand(1, 24))
        b.copy_(rand(1, 24))
        before = bridge.collective_launch_count() if bridge is not None else 0
        outputs = tuple(fn(*inputs) for inputs in calls)
        torch.hpu.synchronize()
        if bridge is not None:
            delta = bridge.collective_launch_count() - before
            assert delta == 6, delta
            launches.append(delta)
        torch.testing.assert_close(outputs[0].cpu(), outputs[1].cpu(), rtol=0, atol=0)
        for offset, start in enumerate((1, 33, 65)):
            torch.testing.assert_close(pool0[start].cpu(), pool1[start].cpu(), rtol=0, atol=0)
            torch.testing.assert_close(conv0[offset].cpu(), conv1[offset].cpu(), rtol=0, atol=0)
        print(f"EXACT_STATE_STEP {step}", flush=True)
    unused = [row for row in range(98) if row not in (1, 33, 65)]
    for pool in (pool0, pool1):
        cpu = pool.cpu()
        torch.testing.assert_close(cpu[unused], initial[unused], rtol=0, atol=0)

    for inputs in calls:
        for _ in range(3):
            fn(*inputs)
        torch.hpu.synchronize()
    device, drained = [[], []], [[], []]
    for wave in range(7):
        for index in ((0, 1) if wave % 2 == 0 else (1, 0)):
            start, end = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                fn(*calls[index])
            end.record()
            torch.hpu.synchronize()
            device[index].append(start.elapsed_time(end) / 20)
    for wave in range(15):
        for index in ((0, 1) if wave % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            fn(*calls[index])
            torch.hpu.synchronize()
            drained[index].append((time.perf_counter() - start) * 1000)
    result = dict(status="pass",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tp2=args.tp2,
                  rank=rank,
                  collective_launch_deltas=launches,
                  torch_version=torch.__version__,
                  stateful_exact_steps=12,
                  inactive_rows_exact=True,
                  pool_shape=list(pool0.shape),
                  view_shape=list(views[0].shape),
                  arms=["whole_pool_input", "external_active_views"],
                  device_ms=device,
                  drained_ms=drained,
                  device_median_ms=list(map(statistics.median, device)),
                  drained_median_ms=list(map(statistics.median, drained)),
                  limitations=[
                      "Three GDN steps plus BF16 projections; not a full model.",
                      "No profiler or NIC overlap measurement."
                  ])
    output_path = args.output.with_name(f"{args.output.stem}-rank{rank}.json") if args.tp2 else args.output
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if args.tp2:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
