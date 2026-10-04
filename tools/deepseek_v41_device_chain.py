# SPDX-License-Identifier: Apache-2.0
"""Native head/frame -> both device Engrams -> ordinary decoder consumer.

Research fixture only: arithmetic and dependencies come from the maintained
serving operators. The certificate is consumed after its provisional consumer
has been queued, as in the ordinary C1 runner. No replacement worker or RPC.
"""
import time

import torch

from vllm_gaudi.ops.deepseek_v41_device_engram import DeviceEngramRounds
from vllm_gaudi.ops.deepseek_v41_device_loop import SamplingFrames

_inputs = {}
_frames = {}
_preparation_observer = None


def set_preparation_observer(callback):
    global _preparation_observer
    _preparation_observer = callback


def close_device_chains():
    for inputs in _inputs.values():
        inputs.close()
    _inputs.clear()
    _frames.clear()


@torch.inference_mode()
def run_device_chain(engine, host, seed_hidden, sampler, full_sample, payloads, bridge,
                     reset, context_tokens, steps, warm_steps, measure, report, *, observer=None):
    def progress(action, index=-1):
        if not measure and _preparation_observer is not None:
            _preparation_observer('device_chain_preparation', action=action, iteration=index)

    progress('reset')
    reset()
    program = engine.program()
    program.sampling_counter.zero_()
    payloads.pop(engine, None)
    progress('initial_sampler')
    initial = sampler(seed_hidden, engine)
    initial_host, initial_done = bridge.copy_sampled_tokens_to_host(initial)
    history = torch.tensor(host.history.history[-3:][::-1].copy(), dtype=torch.int32, device='hpu')
    position = torch.tensor([context_tokens], dtype=torch.int32, device='hpu')
    if engine not in _inputs:
        _inputs[engine] = DeviceEngramRounds(host, initial.reshape(-1), history)
    inputs = _inputs[engine]
    owner = ('native-chain', id(engine))
    torch.hpu.synchronize()
    initial_done.synchronize()
    output_tokens = [int(initial_host[0, 0])]
    delivery = []
    begin, end = ((torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                  if measure else (None, None))
    started = None
    step_events = [(torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                   for _ in range(steps)] if measure else ()
    try:
        current, current_position, current_history = initial, position, history
        certificate = frame = None
        for index in range(warm_steps + steps):
            if measure and index == warm_steps:
                torch.hpu.synchronize()
                started = time.perf_counter_ns()
                begin.record()
            if measure and index >= warm_steps:
                step_events[index-warm_steps][0].record()
            # Producer and real downstream decoder are enqueued before the
            # certificate's native readback is awaited by this host fixture.
            progress('input_producer', index)
            rows = inputs.prepare(owner, current.reshape(-1), current_history)
            progress('decoder_submit', index)
            hidden = engine.from_input_ids(current_position, current.reshape(-1), rows)[0]
            progress('decoder_submitted', index)
            if certificate is not None:
                host_status, done = certificate
                progress('certificate_wait', index)
                done.synchronize()
                covered = bool(int(host_status[0, 0]) & 1)
                report['bounded_sampler_steps'] = report.get('bounded_sampler_steps', 0) + 1
                if covered:
                    token = int(host_status[0, 0]) >> 1
                else:
                    # Same-position repair, with the saved logits and draw.
                    # Discard the provisional completion before reusing roots.
                    engine._complete_input_variant().metadata.native_completion.synchronize()
                    corrected = full_sample(frame[1], frame[2])
                    frame[3].copy_(corrected)
                    rows = inputs.prepare(owner, frame[3].reshape(-1), frame[5])
                    hidden = engine.from_input_ids(frame[4], frame[3].reshape(-1), rows)[0]
                    token = int(corrected.cpu().reshape(-1)[0])
                    report['bounded_sampler_fallbacks'] = report.get('bounded_sampler_fallbacks', 0) + 1
                output_tokens.append(token)
            if measure and index >= warm_steps:
                delivery.append(time.perf_counter_ns())
            if observer is not None:
                progress('hidden_readback', index)
                engine._complete_input_variant().metadata.native_completion.synchronize()
                observer(program, hidden, context_tokens + index)
            progress('sampler', index)
            sampler(hidden, engine)
            payload = payloads.pop(engine)
            if engine not in _frames:
                _frames[engine] = SamplingFrames(payload, inputs.histories[1])
            progress('preserve_frame', index)
            frame = _frames[engine].preserve(payload, inputs.histories[1])
            certificate = bridge.copy_sampled_tokens_to_host(frame[0])
            current, current_position, current_history = frame[3], frame[4], frame[5]
            if measure and index >= warm_steps:
                step_events[index-warm_steps][1].record()
        if measure:
            end.record()
        progress('final_certificate_wait')
        host_status, done = certificate
        done.synchronize()
        if int(host_status[0, 0]) & 1:
            token = int(host_status[0, 0]) >> 1
        else:
            corrected = full_sample(frame[1], frame[2])
            token = int(corrected.cpu().reshape(-1)[0])
            report['bounded_sampler_fallbacks'] = report.get('bounded_sampler_fallbacks', 0) + 1
        report['bounded_sampler_steps'] = report.get('bounded_sampler_steps', 0) + 1
        output_tokens.append(token)
        if measure:
            end.synchronize()
            delivery.append(time.perf_counter_ns())
            elapsed = (time.perf_counter_ns() - started) / 1e6 / steps
            report['token_delivery_ns'] = delivery
            report['device_step_ms'] = [begin.elapsed_time(end) for begin, end in step_events]
            report['checked_first_device_position'] = context_tokens
            device_ms = begin.elapsed_time(end) / steps
        else:
            torch.hpu.synchronize()
            elapsed = device_ms = None
        # The host mirror is a lifecycle check, outside the device interval.
        # It performs no table lookup or transfer and consumes the same IDs.
        for start in range(0, len(output_tokens) - 1, 6):
            committed = output_tokens[start:min(start + 6, len(output_tokens) - 1)]
            batch = host.history.prepare_mirror('chain', committed, [False] * len(committed))
            host.history.commit(batch, len(committed))
        return output_tokens, elapsed, device_ms
    finally:
        if engine.input_variant_ready(program.search_length):
            engine._complete_input_variant().metadata.native_completion.synchronize()
        else:
            torch.hpu.synchronize()
        if inputs.owner == owner:
            inputs.retire(owner)
