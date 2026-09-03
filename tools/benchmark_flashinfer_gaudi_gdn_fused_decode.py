# SPDX-License-Identifier: Apache-2.0
"""Benchmark Gaudi fused-step candidates against the current Qwen GDN chain."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from flashinfer_gaudi._reference import packed_recurrent_decode
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating

_HIDDEN_SIZE = 5120
_QK_HEADS = 16
_VALUE_HEADS = 48
_DIM = 128
_QK_WIDTH = 2 * _QK_HEADS * _DIM
_PACKED_WIDTH = _QK_WIDTH + _VALUE_HEADS * _DIM
_HEAD_REPEAT = _VALUE_HEADS // _QK_HEADS


def _make_inputs(batch: int, with_conv_bias: bool) -> tuple[torch.Tensor | None, ...]:
    generator = torch.Generator().manual_seed(911 + batch)
    hidden = torch.randn(batch, _HIDDEN_SIZE, dtype=torch.bfloat16, generator=generator) * 0.1
    w_ba = torch.randn(2 * _VALUE_HEADS, _HIDDEN_SIZE, dtype=torch.bfloat16, generator=generator) * 0.02
    packed = torch.randn(batch, _PACKED_WIDTH, dtype=torch.bfloat16, generator=generator) * 0.1
    conv_state = torch.randn(batch, 3, _PACKED_WIDTH, dtype=torch.bfloat16, generator=generator) * 0.1
    conv_weight = torch.randn(_PACKED_WIDTH, 4, dtype=torch.bfloat16, generator=generator) * 0.05
    conv_bias = (torch.randn(_PACKED_WIDTH, dtype=torch.bfloat16, generator=generator) *
                 0.01 if with_conv_bias else None)
    A_log = torch.randn(_VALUE_HEADS, dtype=torch.float32, generator=generator) * 0.1 - 2.0
    dt_bias = torch.randn(_VALUE_HEADS, dtype=torch.bfloat16, generator=generator) * 0.1
    ssm_state = torch.randn(batch, _VALUE_HEADS, _DIM, _DIM, dtype=torch.float32, generator=generator) * 0.01
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32)
    return tuple(
        tensor.to("hpu") if tensor is not None else None for tensor in (
            hidden,
            w_ba,
            packed,
            conv_state,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            ssm_state,
            query_start_loc,
        ))


def _gate_to_decay(ba: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, a = ba.chunk(2, dim=-1)
    x = a.float() + dt_bias.float()
    softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    decay = torch.exp(-torch.exp(A_log.float()) * softplus)
    beta = torch.sigmoid(b.float()).to(b.dtype)
    return decay, beta


def _direct_conv(
    packed: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
) -> torch.Tensor:
    weights = conv_weight.float()
    convolved = conv_state[:, 0, :] * weights[:, 0]
    convolved = convolved + conv_state[:, 1, :] * weights[:, 1]
    convolved = convolved + conv_state[:, 2, :] * weights[:, 2]
    convolved = convolved + packed * weights[:, 3]
    if conv_bias is not None:
        convolved = convolved + conv_bias.float()
    convolved = convolved.to(torch.bfloat16)
    output = F.silu(convolved)
    conv_state.copy_(torch.cat((conv_state[:, 1:, :], packed.unsqueeze(1)), dim=1))
    return output


def _normalize_packed_qk(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = packed.shape[0]
    qk = packed[:, :_QK_WIDTH].reshape(batch, 2 * _QK_HEADS, _DIM).float()
    squared_norm = torch.sum(qk * qk, dim=-1, keepdim=True)
    qk = qk * torch.rsqrt(torch.clamp_min(squared_norm, 1e-12))
    q, k = qk.split(_QK_HEADS, dim=1)
    value = packed[:, _QK_WIDTH:].reshape(batch, _QK_HEADS, _HEAD_REPEAT, _DIM).float()
    return q, k, value


def _recurrent_out_of_place(
    packed: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
) -> torch.Tensor:
    batch = packed.shape[0]
    q, k, value = _normalize_packed_qk(packed)
    k = k.unsqueeze(2)
    state = ssm_state.reshape(batch, _QK_HEADS, _HEAD_REPEAT, _DIM, _DIM)
    decay = decay.reshape(batch, _QK_HEADS, _HEAD_REPEAT, 1, 1)
    beta = beta.reshape(batch, _QK_HEADS, _HEAD_REPEAT).float().unsqueeze(-1)
    decayed_state = state * decay
    projection = torch.matmul(decayed_state, k.unsqueeze(-1)).squeeze(-1)
    delta = (value - projection) * beta
    updated_state = torch.addcmul(decayed_state, delta.unsqueeze(-1), k.unsqueeze(-2))
    output = torch.matmul(updated_state, (q.unsqueeze(2) * (_DIM**-0.5)).unsqueeze(-1)).squeeze(-1)
    ssm_state.copy_(updated_state.reshape_as(ssm_state))
    return output.reshape(batch, _VALUE_HEADS, _DIM).to(packed.dtype)


def _recurrent_in_place(
    packed: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
) -> torch.Tensor:
    batch = packed.shape[0]
    q, k, value = _normalize_packed_qk(packed)
    k = k.unsqueeze(2)
    state = ssm_state.reshape(batch, _QK_HEADS, _HEAD_REPEAT, _DIM, _DIM)
    decay = decay.reshape(batch, _QK_HEADS, _HEAD_REPEAT, 1, 1)
    beta = beta.reshape(batch, _QK_HEADS, _HEAD_REPEAT).float().unsqueeze(-1)
    state.mul_(decay)
    projection = torch.matmul(state, k.unsqueeze(-1)).squeeze(-1)
    delta = (value - projection) * beta
    state.addcmul_(delta.unsqueeze(-1), k.unsqueeze(-2))
    output = torch.matmul(state, (q.unsqueeze(2) * (_DIM**-0.5)).unsqueeze(-1)).squeeze(-1)
    return output.reshape(batch, _VALUE_HEADS, _DIM).to(packed.dtype)


def _current_chain(
    hidden: torch.Tensor,
    w_ba: torch.Tensor,
    packed: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    ba = F.linear(hidden, w_ba)
    b, a = ba.chunk(2, dim=-1)
    log_decay, beta = hpu_fused_gdn_gating(A_log, a.contiguous(), b.contiguous(), dt_bias)
    convolved = hpu_causal_conv1d_update(
        x=packed,
        conv_state=conv_state,
        weight=conv_weight,
        bias=conv_bias,
        activation="silu",
        query_start_loc=query_start_loc,
        direct_state_layout=True,
    )
    output, _ = packed_recurrent_decode(
        convolved,
        log_decay,
        beta,
        ssm_state,
        None,
        None,
        _DIM**-0.5,
        True,
        True,
    )
    return output


def _legacy_chain(*inputs: torch.Tensor) -> torch.Tensor:
    hidden, w_ba, packed, conv_state, conv_weight, conv_bias, A_log, dt_bias, ssm_state, query_start_loc = inputs
    ba = F.linear(hidden, w_ba)
    b, a = ba.chunk(2, dim=-1)
    log_decay, beta = hpu_fused_gdn_gating(A_log, a.contiguous(), b.contiguous(), dt_bias)
    convolved = hpu_causal_conv1d_update(
        x=packed,
        conv_state=conv_state,
        weight=conv_weight,
        bias=conv_bias,
        activation="silu",
        query_start_loc=query_start_loc,
        direct_state_layout=True,
    )
    decay = torch.exp(log_decay.reshape(packed.shape[0], _VALUE_HEADS))
    return _recurrent_out_of_place(convolved, decay, beta, ssm_state)


def _fused_out_of_place(*inputs: torch.Tensor) -> torch.Tensor:
    hidden, w_ba, packed, conv_state, conv_weight, conv_bias, A_log, dt_bias, ssm_state, query_start_loc = inputs
    del query_start_loc
    decay, beta = _gate_to_decay(F.linear(hidden, w_ba), A_log, dt_bias)
    convolved = _direct_conv(packed, conv_state, conv_weight, conv_bias)
    return _recurrent_out_of_place(convolved, decay, beta, ssm_state)


def _fused_in_place(*inputs: torch.Tensor) -> torch.Tensor:
    hidden, w_ba, packed, conv_state, conv_weight, conv_bias, A_log, dt_bias, ssm_state, query_start_loc = inputs
    del query_start_loc
    decay, beta = _gate_to_decay(F.linear(hidden, w_ba), A_log, dt_bias)
    convolved = _direct_conv(packed, conv_state, conv_weight, conv_bias)
    return _recurrent_in_place(convolved, decay, beta, ssm_state)


def _inplace_recurrence_only(*inputs: torch.Tensor) -> torch.Tensor:
    hidden, w_ba, packed, conv_state, conv_weight, conv_bias, A_log, dt_bias, ssm_state, query_start_loc = inputs
    ba = F.linear(hidden, w_ba)
    b, a = ba.chunk(2, dim=-1)
    log_decay, beta = hpu_fused_gdn_gating(A_log, a.contiguous(), b.contiguous(), dt_bias)
    convolved = hpu_causal_conv1d_update(
        x=packed,
        conv_state=conv_state,
        weight=conv_weight,
        bias=conv_bias,
        activation="silu",
        query_start_loc=query_start_loc,
        direct_state_layout=True,
    )
    return _recurrent_in_place(convolved, torch.exp(log_decay.reshape(packed.shape[0], _VALUE_HEADS)), beta, ssm_state)


def _direct_decay_current_conv(*inputs: torch.Tensor) -> torch.Tensor:
    hidden, w_ba, packed, conv_state, conv_weight, conv_bias, A_log, dt_bias, ssm_state, query_start_loc = inputs
    decay, beta = _gate_to_decay(F.linear(hidden, w_ba), A_log, dt_bias)
    convolved = hpu_causal_conv1d_update(
        x=packed,
        conv_state=conv_state,
        weight=conv_weight,
        bias=conv_bias,
        activation="silu",
        query_start_loc=query_start_loc,
        direct_state_layout=True,
    )
    return _recurrent_out_of_place(convolved, decay, beta, ssm_state)


def _direct_decay_current_conv_inplace(*inputs: torch.Tensor) -> torch.Tensor:
    hidden, w_ba, packed, conv_state, conv_weight, conv_bias, A_log, dt_bias, ssm_state, query_start_loc = inputs
    decay, beta = _gate_to_decay(F.linear(hidden, w_ba), A_log, dt_bias)
    convolved = hpu_causal_conv1d_update(
        x=packed,
        conv_state=conv_state,
        weight=conv_weight,
        bias=conv_bias,
        activation="silu",
        query_start_loc=query_start_loc,
        direct_state_layout=True,
    )
    return _recurrent_in_place(convolved, decay, beta, ssm_state)


def _current_gate_manual_conv(*inputs: torch.Tensor) -> torch.Tensor:
    hidden, w_ba, packed, conv_state, conv_weight, conv_bias, A_log, dt_bias, ssm_state, query_start_loc = inputs
    del query_start_loc
    ba = F.linear(hidden, w_ba)
    b, a = ba.chunk(2, dim=-1)
    log_decay, beta = hpu_fused_gdn_gating(A_log, a.contiguous(), b.contiguous(), dt_bias)
    convolved = _direct_conv(packed, conv_state, conv_weight, conv_bias)
    decay = torch.exp(log_decay.reshape(packed.shape[0], _VALUE_HEADS))
    return _recurrent_out_of_place(convolved, decay, beta, ssm_state)


def _current_gate_manual_conv_inplace(*inputs: torch.Tensor) -> torch.Tensor:
    hidden, w_ba, packed, conv_state, conv_weight, conv_bias, A_log, dt_bias, ssm_state, query_start_loc = inputs
    del query_start_loc
    ba = F.linear(hidden, w_ba)
    b, a = ba.chunk(2, dim=-1)
    log_decay, beta = hpu_fused_gdn_gating(A_log, a.contiguous(), b.contiguous(), dt_bias)
    convolved = _direct_conv(packed, conv_state, conv_weight, conv_bias)
    decay = torch.exp(log_decay.reshape(packed.shape[0], _VALUE_HEADS))
    return _recurrent_in_place(convolved, decay, beta, ssm_state)


def _compile(function):
    return torch.compile(function, backend="hpu_backend", fullgraph=True, dynamic=False)


def _measure_wave(function, inputs: tuple[torch.Tensor | None, ...], iterations: int) -> tuple[float, float]:
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function(*inputs)
    end_event.record()
    torch.hpu.synchronize()
    queued_ms = (time.perf_counter_ns() - started) / 1e6 / iterations
    return queued_ms, start_event.elapsed_time(end_event) / iterations


def _statistics(samples: list[tuple[float, float]]) -> dict[str, float]:
    return {
        "queued_median_ms": statistics.median(sample[0] for sample in samples),
        "device_median_ms": statistics.median(sample[1] for sample in samples),
        "device_min_ms": min(sample[1] for sample in samples),
    }


def _quality(
    reference_output: torch.Tensor,
    reference_inputs: tuple[torch.Tensor | None, ...],
    output: torch.Tensor,
    inputs: tuple[torch.Tensor | None, ...],
) -> dict[str, float]:
    return {
        "output_max_abs": float((output.float() - reference_output.float()).abs().max().cpu()),
        "conv_state_max_abs": float((inputs[3].float() - reference_inputs[3].float()).abs().max().cpu()),
        "ssm_state_max_abs": float((inputs[8] - reference_inputs[8]).abs().max().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4,8,12,16,20,32")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--wave-iterations", type=int, default=100)
    parser.add_argument("--waves", type=int, default=7)
    parser.add_argument("--variants", default="current,fused_out_of_place")
    parser.add_argument("--reference", default="current")
    parser.add_argument("--with-conv-bias", action="store_true")
    args = parser.parse_args()

    results: dict[str, object] = {"batches": {}}
    available_functions = {
        "legacy": _legacy_chain,
        "current": _current_chain,
        "fused_out_of_place": _fused_out_of_place,
        "fused_in_place": _fused_in_place,
        "inplace_recurrence_only": _inplace_recurrence_only,
        "direct_decay_current_conv": _direct_decay_current_conv,
        "direct_decay_current_conv_inplace": _direct_decay_current_conv_inplace,
        "current_gate_manual_conv": _current_gate_manual_conv,
        "current_gate_manual_conv_inplace": _current_gate_manual_conv_inplace,
    }
    variant_names = tuple(item for item in args.variants.split(",") if item)
    if args.reference not in variant_names:
        raise ValueError("--variants must include the selected --reference.")
    unknown = set(variant_names) - set(available_functions)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    functions = {name: available_functions[name] for name in variant_names}
    for batch in (int(item) for item in args.batches.split(",") if item):
        torch._dynamo.reset()
        base_inputs = _make_inputs(batch, args.with_conv_bias)
        inputs_by_name = {
            name: tuple(tensor.clone() if tensor is not None else None for tensor in base_inputs)
            for name in functions
        }
        compiled = {name: _compile(function) for name, function in functions.items()}

        outputs = {name: function(*inputs_by_name[name]) for name, function in compiled.items()}
        torch.hpu.synchronize()
        reference_output = outputs[args.reference]
        reference_inputs = inputs_by_name[args.reference]
        quality = {
            name: _quality(reference_output, reference_inputs, output, inputs_by_name[name])
            for name, output in outputs.items() if name != args.reference
        }

        for _ in range(args.warmups):
            for name, function in compiled.items():
                function(*inputs_by_name[name])
        torch.hpu.synchronize()

        samples: dict[str, list[tuple[float, float]]] = {name: [] for name in functions}
        names = tuple(functions)
        for wave in range(args.waves):
            offset = wave % len(names)
            ordered = names[offset:] + names[:offset]
            for name in ordered:
                samples[name].append(_measure_wave(compiled[name], inputs_by_name[name], args.wave_iterations))
        stats = {name: _statistics(values) for name, values in samples.items()}
        reference_ms = stats[args.reference]["device_median_ms"]
        results["batches"][str(batch)] = {
            "stats": stats,
            "quality": quality,
            "device_speedup": {
                name: reference_ms / values["device_median_ms"]
                for name, values in stats.items() if name != args.reference
            },
        }
        print(json.dumps({str(batch): results["batches"][str(batch)]}, indent=2, sort_keys=True), flush=True)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
