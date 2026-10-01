# SPDX-License-Identifier: Apache-2.0
"""Qualify input DMA -> changing integer producer -> D2H -> next input reuse."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-inputs', action='store_true')
    parser.add_argument('--saved-reference', type=Path)
    args = parser.parse_args()
    if args.native_inputs and not args.saved_reference:
        parser.error('Native input qualification must reuse the archived control-chain reference')
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment('/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2',
                        tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch  # noqa: F401
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_inputs import PinnedDecodeInputs
    from vllm_gaudi.ops.deepseek_v41_completion import prepare_token_readback
    torch.hpu.set_device(0)
    bind_worker_cpu(0)
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(status='running', scope='One-card control chain; not TP4 model or HBM bandwidth qualification', cases=[])
    try:
        with torch.inference_mode():
            staging = PinnedDecodeInputs(6, 'hpu:0')
            readback = prepare_token_readback()
            native = None
            if args.native_inputs:
                from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge
                bridge = _load_bridge(Path(os.environ['VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE']).resolve())
                assert bridge.prepared_control_input_version == 1

            def producer(ids, positions):
                return ((ids + positions.to(torch.int64)) * 13 + 17).sum().reshape(1, 1)
            compiled = torch.compile(producer, backend='hpu_backend', fullgraph=True, dynamic=False)
            frames = {count: (torch.empty(count, dtype=torch.int64, device='hpu'),
                              torch.empty(count, dtype=torch.int32, device='hpu')) for count in (1, 2, 6)}
            if args.native_inputs:
                native = {count: bridge.PreparedControlInputs(*values) for count, values in frames.items()}
            for count, (ids, positions) in frames.items():
                if native is None:
                    staging.stage(list(range(count)), 16384, ids, positions)
                else:
                    native[count].upload(list(range(count)), 16384)
                host, event = readback(compiled(ids, positions))
                event.synchronize()
                assert int(host[0, 0]) == sum((value + 16384 + value) * 13 + 17 for value in range(count))
            bind_worker_helpers(0)
            if args.saved_reference:
                saved = json.loads(args.saved_reference.read_text())
                assert saved['status'] == 'passed'
                report['reference'] = str(args.saved_reference.resolve())
                modes = ('candidate',)
            else:
                # The first run collects the missing component reference once.
                modes = ('reference', 'candidate')
            for count in (1, 2, 6):
                ids, positions = frames[count]
                for mode in modes:
                    previous = 7
                    samples = []
                    steps = []
                    for step in range(48):
                        start = 16384 + step * count
                        values = [(previous + i) % 129280 for i in range(count)]
                        expected = sum((value + start + i) * 13 + 17 for i, value in enumerate(values))
                        began = time.perf_counter_ns()
                        if mode == 'reference':
                            ids.copy_(torch.tensor(values, dtype=torch.int64))
                            positions.copy_(torch.arange(start, start + count, dtype=torch.int32))
                        elif native is None:
                            staging.stage(values, start, ids, positions)
                        else:
                            native[count].upload(values, start)
                        staged = time.perf_counter_ns()
                        selected = compiled(ids, positions)
                        submitted = time.perf_counter_ns()
                        if mode == 'reference':
                            actual = int(selected.cpu()[0, 0])
                        else:
                            host, done = readback(selected)
                            done.synchronize()
                            actual = int(host[0, 0])
                        end = time.perf_counter_ns()
                        assert actual == expected, (count, mode, step, actual, expected)
                        # The real host consumer drives the subsequent DMA and
                        # compiled producer; the wait cannot be hidden past it.
                        previous = actual
                        samples.append((end - began) / 1e6)
                        steps.append(dict(staging_ms=(staged-began)/1e6, submit_ms=(submitted-staged)/1e6,
                                          consume_ms=(end-submitted)/1e6))
                    report['cases'].append(dict(tokens=count, mode=mode, exact=True, samples_ms=samples,
                                                median_ms=statistics.median(samples), steps=steps))
                    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
            if native is not None:
                # Multiple users of one destination must stay ordered even
                # when the host does not consume each result immediately.
                queued = []
                for step in range(24):
                    count = (1, 2, 6)[step % 3]
                    values = [step * 7 + index for index in range(count)]
                    start = 20000 + step * 6
                    native[count].upload(values, start)
                    host, done = readback(compiled(*frames[count]))
                    queued.append((host, done, sum((v + start + i) * 13 + 17 for i, v in enumerate(values))))
                for host, done, expected in queued:
                    done.synchronize()
                    assert int(host[0, 0]) == expected
                rejected = 0
                for values, start in (([], 0), ([1], -1), ([1], 2**31)):
                    try:
                        native[1].upload(values, start)
                    except RuntimeError:
                        rejected += 1
                assert rejected == 3
                if native[1].payload_bytes == 12:
                    native[1].upload([2**40], 0)
                    # Test the upload's physical width directly. The synthetic
                    # HPU integer arithmetic chain is not a general int64 ALU
                    # reference for values far outside the model vocabulary.
                    assert int(frames[1][0].cpu()[0]) == 2**40
                    report['int64_roundtrip'] = True
                else:
                    try:
                        native[1].upload([2**40], 0)
                    except RuntimeError:
                        rejected += 1
                    assert rejected == 4
                rejected_frames = 0
                for ids, pos in ((torch.empty(1, device='hpu'), frames[1][1]),
                                 (frames[1][0], torch.empty(1, dtype=torch.int64, device='hpu')),
                                 (frames[2][0][1:], frames[1][1])):
                    try:
                        bridge.PreparedControlInputs(ids, pos)
                    except RuntimeError:
                        rejected_frames += 1
                assert rejected_frames == 3
                report.update(queued_reuse_exact=len(queued), invalid_uploads_rejected=rejected,
                              invalid_frames_rejected=rejected_frames,
                              native_inputs={n: dict(uploads=v.uploads, payload_bytes=v.payload_bytes)
                                             for n, v in native.items()})
            report.update(status='passed', pinned_waits=staging.waits, logical_input_bytes=staging.bytes)
    except Exception as error:
        import traceback
        traceback.print_exc()
        report.update(status='failed', error=repr(error))
    finally:
        (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'cases'}), flush=True)
        if report['status'] != 'passed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
