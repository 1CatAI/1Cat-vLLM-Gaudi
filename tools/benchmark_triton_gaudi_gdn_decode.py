# SPDX-License-Identifier: Apache-2.0
"""Gaudi2 performance gate for fused Triton Qwen3.5 GDN decode."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable

from triton.backends.gaudi.driver import prepare_environment

prepare_environment()
os.environ.setdefault("VLLM_HPU_TRITON_MODE", "strict")

import torch  # noqa: E402

from vllm_gaudi.ops.triton_gaudi import (  # noqa: E402
    gdn_decode_conv_split_packed, gdn_decode_packed, prepare_if_enabled,
)
from vllm_gaudi.ops.triton_gaudi.kernels import (  # noqa: E402
    gdn_decode_conv_split_packed_direct, )
from vllm_gaudi.ops.causal_conv1d_pytorch import (  # noqa: E402
    hpu_causal_conv1d_update, )

KEY_HEADS = 16
VALUE_HEADS = 48
KEY_DIM = 128
VALUE_DIM = 128
PACKED_WIDTH = 10240


def _reference(
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    batch = packed_qkv.shape[0]
    q_size = KEY_HEADS * KEY_DIM
    query, key, value = packed_qkv.split(
        (q_size, q_size, VALUE_HEADS * VALUE_DIM),
        dim=-1,
    )
    query = query.float().view(batch, KEY_HEADS, KEY_DIM)
    key = key.float().view(batch, KEY_HEADS, KEY_DIM)
    query = query * torch.rsqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1.0e-6)
    query = query * (KEY_DIM**-0.5)
    key = key * torch.rsqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1.0e-6)
    query = query.repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=1)
    key = key.repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=1)
    value = value.float().view(batch, VALUE_HEADS, VALUE_DIM)

    gate_x = gate_a.float() + dt_bias
    softplus = torch.where(
        gate_x <= 20.0,
        torch.log1p(torch.exp(gate_x)),
        gate_x,
    )
    decay = torch.exp(-torch.exp(a_log) * softplus)
    beta = torch.sigmoid(gate_b.float())
    safe_indices = torch.remainder(state_indices.long(), state_cache.shape[0])
    state = state_cache.index_select(0, safe_indices)
    state = state * decay.unsqueeze(-1).unsqueeze(-1)
    projection = torch.sum(state * key.unsqueeze(-2), dim=-1)
    delta = (value - projection) * beta.unsqueeze(-1)
    state = state + delta.unsqueeze(-1) * key.unsqueeze(-2)
    output = torch.sum(state * query.unsqueeze(-2), dim=-1).bfloat16()
    state_cache.index_copy_(0, safe_indices, state)
    return output


def _triton(
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    output = gdn_decode_packed(
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
    )
    if output is None:
        raise RuntimeError("strict Gaudi Triton benchmark unexpectedly selected the vendor path")
    return output


def _reference_with_conv(
    conv_state: torch.Tensor,
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_weight_t: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    packed_qkv_conv = hpu_causal_conv1d_update(
        x=packed_qkv,
        conv_state=conv_state,
        weight=conv_weight,
        bias=None,
        activation="silu",
        conv_state_indices=state_indices,
        query_start_loc=query_start_loc,
    )
    return _reference(
        state_cache,
        packed_qkv_conv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
    )


def _triton_with_conv(
    conv_state: torch.Tensor,
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_weight_t: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    del conv_weight, query_start_loc
    output = gdn_decode_conv_split_packed(
        conv_state,
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
    )
    if output is None:
        raise RuntimeError("strict Gaudi Triton benchmark unexpectedly rejected fused conv+GDN")
    return output


def _triton_with_conv_direct(
    conv_state: torch.Tensor,
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_weight_t: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    del conv_weight, query_start_loc
    return gdn_decode_conv_split_packed_direct(
        conv_state,
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
    )


def _timed(
    function: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    repetitions: int,
) -> tuple[float, float]:
    start = torch.hpu.Event(enable_timing=True)
    end = torch.hpu.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter_ns()
    output = None
    for _ in range(repetitions):
        output = function(*inputs)
    end.record()
    end.synchronize()
    if output is None:
        raise AssertionError("benchmark did not execute")
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000 / repetitions
    return start.elapsed_time(end) / repetitions, wall_ms


def _benchmark_pair(
    vendor: Callable[..., torch.Tensor],
    triton_kernel: Callable[..., torch.Tensor],
    vendor_inputs: tuple[torch.Tensor, ...],
    triton_inputs: tuple[torch.Tensor, ...],
    warmup: int,
    repetitions: int,
    rounds: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    for _ in range(warmup):
        vendor(*vendor_inputs)
        triton_kernel(*triton_inputs)
    torch.hpu.synchronize()
    samples: dict[str, list[tuple[float, float]]] = {
        "vendor": [],
        "triton": [],
    }
    functions = {
        "vendor": (vendor, vendor_inputs),
        "triton": (triton_kernel, triton_inputs),
    }
    for round_index in range(rounds):
        order = (("vendor", "triton") if round_index % 2 == 0 else (
            "triton",
            "vendor",
        ))
        for name in order:
            function, inputs = functions[name]
            samples[name].append(_timed(function, inputs, repetitions))
    medians = {
        name: (
            statistics.median(sample[0] for sample in values),
            statistics.median(sample[1] for sample in values),
        )
        for name, values in samples.items()
    }
    return medians["vendor"], medians["triton"]


def _check_correctness(
    vendor: Callable[..., torch.Tensor],
    triton_kernel: Callable[..., torch.Tensor],
    vendor_inputs: tuple[torch.Tensor, ...],
    triton_inputs: tuple[torch.Tensor, ...],
    steps: int,
    state_arg: int,
    conv_state_arg: int | None,
    include_conv: bool,
) -> dict[str, float | int | None]:
    initial_vendor_state = vendor_inputs[state_arg].clone()
    initial_state = triton_inputs[state_arg].clone()
    initial_conv_state = triton_inputs[conv_state_arg].clone() if conv_state_arg is not None else None
    expected_output = None
    actual_output = None
    for _ in range(steps):
        expected_output = vendor(*vendor_inputs)
        actual_output = triton_kernel(*triton_inputs)
    torch.hpu.synchronize()
    assert expected_output is not None and actual_output is not None
    output_max_abs = (actual_output.float() - expected_output.float()).abs().max().cpu().item()
    state_difference = triton_inputs[state_arg] - vendor_inputs[state_arg]
    state_max_abs = state_difference.abs().max().cpu().item()
    state_rmse = torch.sqrt(torch.mean(state_difference * state_difference)).cpu().item()
    state_mutation_max_abs = (triton_inputs[state_arg] - initial_state).abs().max().cpu().item()
    vendor_state_mutation_max_abs = (vendor_inputs[state_arg] - initial_vendor_state).abs().max().cpu().item()
    conv_state_max_abs = None
    conv_state_mutation_max_abs = None
    if conv_state_arg is not None:
        conv_state_max_abs = (triton_inputs[conv_state_arg] - vendor_inputs[conv_state_arg]).abs().max().cpu().item()
        conv_state_mutation_max_abs = (triton_inputs[conv_state_arg] - initial_conv_state).abs().max().cpu().item()

    torch.testing.assert_close(
        actual_output,
        expected_output,
        rtol=0.025,
        atol=0.002,
    )
    if include_conv:
        if state_max_abs > 4.0e-3 or state_rmse > 5.0e-5:
            raise AssertionError("fused conv+GDN recurrent-state drift exceeded the "
                                 "bounded stability envelope: "
                                 f"max_abs={state_max_abs}, rmse={state_rmse}, "
                                 f"mutation_max_abs={state_mutation_max_abs}, "
                                 f"vendor_mutation_max_abs={vendor_state_mutation_max_abs}")
    else:
        torch.testing.assert_close(
            triton_inputs[state_arg],
            vendor_inputs[state_arg],
            rtol=1.0e-5,
            atol=2.0e-6,
        )
    if conv_state_arg is not None:
        torch.testing.assert_close(
            triton_inputs[conv_state_arg],
            vendor_inputs[conv_state_arg],
            rtol=0.0,
            atol=0.0,
        )
    return {
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
        "state_rmse": state_rmse,
        "state_mutation_max_abs": state_mutation_max_abs,
        "vendor_state_mutation_max_abs": vendor_state_mutation_max_abs,
        "conv_state_max_abs": conv_state_max_abs,
        "conv_state_mutation_max_abs": conv_state_mutation_max_abs,
        "steps": steps,
    }


def _make_cpu_inputs(
    batch: int,
    state_slots: int,
    state_index_offset: int,
    include_conv: bool,
) -> tuple[torch.Tensor, ...]:
    core_inputs = (
        torch.randn(
            state_slots,
            VALUE_HEADS,
            VALUE_DIM,
            KEY_DIM,
            dtype=torch.float32,
        ) * 0.01,
        (torch.randn(batch, PACKED_WIDTH) * 0.05).bfloat16(),
        (torch.randn(batch, VALUE_HEADS) * 0.2).bfloat16(),
        (torch.randn(batch, VALUE_HEADS) * 0.2).bfloat16(),
        torch.randn(VALUE_HEADS, dtype=torch.float32) * 0.05,
        torch.randn(VALUE_HEADS, dtype=torch.float32) * 0.05,
        torch.arange(batch, dtype=torch.int32).flip(0) + state_index_offset,
    )
    if not include_conv:
        return core_inputs
    conv_weight = (torch.randn(PACKED_WIDTH, 4) * 0.02).bfloat16()
    return (
        (torch.randn(state_slots, 3, PACKED_WIDTH) * 0.01).bfloat16(),
        *core_inputs,
        conv_weight,
        conv_weight.transpose(0, 1).contiguous(),
        torch.arange(batch + 1, dtype=torch.int32),
    )


def _check_recipe_reentry(
    vendor: Callable[..., torch.Tensor],
    triton_kernel: Callable[..., torch.Tensor],
    vendor_inputs: tuple[torch.Tensor, ...],
    triton_inputs: tuple[torch.Tensor, ...],
    interloper: Callable[..., torch.Tensor],
    vendor_interloper_inputs: tuple[torch.Tensor, ...],
    triton_interloper_inputs: tuple[torch.Tensor, ...],
    steps: int,
    state_arg: int,
    conv_state_arg: int,
) -> dict[str, float | int | None]:

    def state_differences() -> dict[str, float]:
        return {
            "conv_state_max_abs":
            (triton_inputs[conv_state_arg] - vendor_inputs[conv_state_arg]).abs().max().cpu().item(),
            "state_max_abs": (triton_inputs[state_arg] - vendor_inputs[state_arg]).abs().max().cpu().item(),
        }

    triton_inputs[state_arg].copy_(vendor_inputs[state_arg])
    triton_inputs[conv_state_arg].copy_(vendor_inputs[conv_state_arg])
    torch.hpu.synchronize()

    for _ in range(steps):
        vendor(*vendor_inputs)
        triton_kernel(*triton_inputs)
    torch.hpu.synchronize()
    print(json.dumps({"recipe_reentry_before_switch": state_differences()}, sort_keys=True))
    for _ in range(steps):
        interloper(*vendor_interloper_inputs)
        interloper(*triton_interloper_inputs)
    torch.hpu.synchronize()
    print(json.dumps({"recipe_reentry_after_switch": state_differences()}, sort_keys=True))

    return _check_correctness(
        vendor,
        triton_kernel,
        vendor_inputs,
        triton_inputs,
        steps,
        state_arg,
        conv_state_arg,
        include_conv=True,
    )


def _check_index_rebind(
    vendor: Callable[..., torch.Tensor],
    triton_kernel: Callable[..., torch.Tensor],
    vendor_inputs: tuple[torch.Tensor, ...],
    triton_inputs: tuple[torch.Tensor, ...],
    steps: int,
    state_arg: int,
    conv_state_arg: int,
    state_indices_arg: int,
) -> dict[str, float | int | None]:
    batch = vendor_inputs[state_indices_arg].numel()
    state_slots = vendor_inputs[state_arg].shape[0]
    if state_slots < batch * 2:
        raise ValueError("index rebind checking requires at least two disjoint batches of state slots")
    offset = state_slots - batch
    rebound_indices = (torch.arange(batch, dtype=torch.int32).flip(0) + offset).to("hpu")
    vendor_rebound = list(vendor_inputs)
    triton_rebound = list(triton_inputs)
    vendor_rebound[state_indices_arg] = rebound_indices
    triton_rebound[state_indices_arg] = rebound_indices.clone()
    return _check_correctness(
        vendor,
        triton_kernel,
        tuple(vendor_rebound),
        tuple(triton_rebound),
        steps,
        state_arg,
        conv_state_arg,
        include_conv=True,
    )


def _install_reinplace_diagnostics() -> None:
    from habana_frameworks.torch.dynamo.compile_backend import passes

    original = passes.pass_reinplace_triton_gaudi_gdn_decode

    def diagnostic(ctx):
        gdn_ops = {
            torch.ops.triton_gaudi.gdn_decode_packed.default,
            torch.ops.triton_gaudi.gdn_decode_conv_packed.default,
            torch.ops.triton_gaudi.gdn_qk_conv_packed.default,
            torch.ops.triton_gaudi.gdn_decode_value_conv_packed.default,
        }
        functionalized = torch.ops.higher_order.auto_functionalized_v2
        for node in ctx.graph_module.graph.nodes:
            if node.op == "call_function" and node.target == functionalized and node.args and node.args[0] in gdn_ops:
                bases = node.kwargs.get("_all_bases", ())
                base_users = [[(user.name, str(user.target))
                               for user in base.users] if isinstance(base, torch.fx.Node) else [] for base in bases]
                print(
                    json.dumps(
                        {
                            "reinplace_candidate": node.name,
                            "op": str(node.args[0]),
                            "kwargs": sorted(node.kwargs),
                            "users": [(user.name, str(user.target)) for user in node.users],
                            "base_users": base_users,
                        },
                        sort_keys=True,
                    ))
        if any(node.op == "call_function" and node.target == functionalized and node.args and node.args[0] in gdn_ops
               for node in ctx.graph_module.graph.nodes):
            print(ctx.graph_module.graph)
        changed = original(ctx)
        print(json.dumps({"reinplace_changed": changed}, sort_keys=True))
        if changed:
            print(ctx.graph_module.graph)
        return changed

    passes.pass_reinplace_triton_gaudi_gdn_decode = diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--state-slots", type=int, default=0)
    parser.add_argument("--state-index-offset", type=int, default=1)
    parser.add_argument("--reentry-batch", type=int, default=32)
    parser.add_argument("--reentry-steps", type=int, default=3)
    parser.add_argument("--value-tile", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--correctness-steps", type=int, default=8)
    parser.add_argument("--min-speedup", type=float, default=1.20)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--skip-eager-correctness", action="store_true")
    parser.add_argument("--direct-triton", action="store_true")
    parser.add_argument(
        "--include-conv",
        action="store_true",
        help="benchmark the real causal-conv -> GDN decode subgraph",
    )
    parser.add_argument("--debug-reinplace", action="store_true")
    parser.add_argument(
        "--check-recipe-reentry",
        action="store_true",
        help="reset state after another batch recipe, then validate re-entry",
    )
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("batch must be positive")
    if args.state_slots < 0:
        parser.error("state-slots cannot be negative")
    if args.state_index_offset < 0:
        parser.error("state-index-offset cannot be negative")
    if args.reentry_batch <= 0 or args.reentry_steps <= 0:
        parser.error("reentry-batch and reentry-steps must be positive")
    if args.value_tile not in (16, 32, 64, 128):
        parser.error("value-tile must be 16, 32, 64, or 128")
    if args.warmup < 1 or args.repetitions < 1 or args.rounds < 1 or args.correctness_steps < 1:
        parser.error("warmup, repetitions, rounds, and correctness-steps must be positive")
    if args.check_recipe_reentry and (args.eager or args.direct_triton or not args.include_conv):
        parser.error("recipe re-entry checking requires compiled --include-conv mode")
    if args.skip_eager_correctness and args.eager:
        parser.error("cannot skip eager correctness in eager benchmark mode")

    state_slots = args.state_slots or args.batch + 3
    if args.state_index_offset + args.batch > state_slots:
        parser.error("state-index-offset plus batch must not exceed state-slots")
    os.environ["VLLM_HPU_TRITON_GDN_VALUE_TILE"] = str(args.value_tile)
    # Resolve a single HABANA_VISIBLE_MODULES entry before artifact
    # registration or any implicit device acquisition occurs.
    torch.hpu.set_device(0)
    if args.debug_reinplace:
        _install_reinplace_diagnostics()
    prepare_if_enabled()
    torch.set_grad_enabled(False)
    torch.manual_seed(20260902)

    cpu_inputs = _make_cpu_inputs(
        args.batch,
        state_slots,
        args.state_index_offset,
        args.include_conv,
    )
    if args.include_conv:
        vendor_impl = _reference_with_conv
        triton_impl = _triton_with_conv_direct if args.direct_triton else _triton_with_conv
        state_arg = 1
        conv_state_arg = 0
    else:
        vendor_impl = _reference
        triton_impl = _triton
        state_arg = 0
        conv_state_arg = None

    vendor_inputs = tuple(tensor.to("hpu") for tensor in cpu_inputs)
    triton_inputs = tuple(tensor.clone() for tensor in vendor_inputs)
    correctness = None
    if not args.skip_eager_correctness:
        correctness = _check_correctness(
            vendor_impl,
            triton_impl,
            vendor_inputs,
            triton_inputs,
            args.correctness_steps,
            state_arg,
            conv_state_arg,
            args.include_conv,
        )
        print(json.dumps({"correctness": correctness}, sort_keys=True))

    if not args.eager:
        vendor_impl = torch.compile(
            vendor_impl,
            backend="hpu_backend",
            fullgraph=True,
            dynamic=False,
        )
        if not args.direct_triton:
            triton_impl = torch.compile(
                triton_impl,
                backend="hpu_backend",
                fullgraph=True,
                dynamic=False,
            )
        vendor_inputs = tuple(tensor.to("hpu") for tensor in cpu_inputs)
        triton_inputs = tuple(tensor.clone() for tensor in vendor_inputs)
        correctness = _check_correctness(
            vendor_impl,
            triton_impl,
            vendor_inputs,
            triton_inputs,
            args.correctness_steps,
            state_arg,
            conv_state_arg,
            args.include_conv,
        )
        print(json.dumps({"compiled_correctness": correctness}, sort_keys=True))

        if args.check_recipe_reentry:
            interloper_slots = max(state_slots, args.reentry_batch + 3)
            interloper_cpu_inputs = _make_cpu_inputs(
                args.reentry_batch,
                interloper_slots,
                1,
                include_conv=True,
            )
            interloper_inputs = tuple(tensor.to("hpu") for tensor in interloper_cpu_inputs)
            vendor_interloper_inputs = (
                vendor_inputs[conv_state_arg],
                vendor_inputs[state_arg],
                *interloper_inputs[2:],
            )
            triton_interloper_inputs = (
                triton_inputs[conv_state_arg],
                triton_inputs[state_arg],
                *interloper_inputs[2:],
            )
            interloper = torch.compile(
                _reference_with_conv,
                backend="hpu_backend",
                fullgraph=True,
                dynamic=False,
            )
            reentry = _check_recipe_reentry(
                vendor_impl,
                triton_impl,
                vendor_inputs,
                triton_inputs,
                interloper,
                vendor_interloper_inputs,
                triton_interloper_inputs,
                args.reentry_steps,
                state_arg,
                conv_state_arg,
            )
            print(json.dumps({"recipe_reentry_correctness": reentry}, sort_keys=True))
            index_rebind = _check_index_rebind(
                vendor_impl,
                triton_impl,
                vendor_inputs,
                triton_inputs,
                args.reentry_steps,
                state_arg,
                conv_state_arg,
                state_indices_arg=7,
            )
            print(json.dumps({"index_rebind_correctness": index_rebind}, sort_keys=True))

    assert correctness is not None
    vendor_times, triton_times = _benchmark_pair(
        vendor_impl,
        triton_impl,
        vendor_inputs,
        triton_inputs,
        args.warmup,
        args.repetitions,
        args.rounds,
    )
    result = {
        "batch": args.batch,
        "state_slots": state_slots,
        "state_index_offset": args.state_index_offset,
        "value_tile": args.value_tile,
        "include_conv": args.include_conv,
        "vendor_torch_compile_fullgraph": not args.eager,
        "triton_torch_compile_fullgraph": (not args.eager and not args.direct_triton),
        "output_max_abs": correctness["output_max_abs"],
        "state_max_abs": correctness["state_max_abs"],
        "state_rmse": correctness["state_rmse"],
        "conv_state_max_abs": correctness["conv_state_max_abs"],
        "vendor_device_ms": vendor_times[0],
        "triton_device_ms": triton_times[0],
        "device_speedup": vendor_times[0] / triton_times[0],
        "vendor_wall_ms": vendor_times[1],
        "triton_wall_ms": triton_times[1],
        "wall_speedup": vendor_times[1] / triton_times[1],
    }
    passed = result["device_speedup"] >= args.min_speedup and result["wall_speedup"] >= args.min_speedup
    print(json.dumps({"passed": passed, "result": result}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
