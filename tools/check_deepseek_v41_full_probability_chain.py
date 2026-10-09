# SPDX-License-Identifier: Apache-2.0
"""Compare exact full p/q materialization through prefix commit, after gather.

Transport is unchanged and excluded from this single-card component. All
probability rows have the real global vocabulary width. This gate cannot
establish Target C6 or complete-round gains.
"""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[0]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from deepseek_v41_micro_replay import RecipeRecorder
    from vllm_gaudi.ops.deepseek_v41_sampling import sample_probabilities
    from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
        filtered_distribution, sample_full_distribution, sample_speculative_prefix)
    from vllm_gaudi.ops.deepseek_v41_verify import verify_control_from_sampled
    from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    recorder = RecipeRecorder(root)
    configure_post_graph_directory(root / 'graphs/rank0')
    vocabulary = json.loads((args.prepared / 'config.json').read_text())['text_config']['vocab_size']

    def body(candidate):
        def distribution(logits, controls):
            if candidate:
                return sample_full_distribution(logits, controls)
            token = sample_probabilities(logits, controls, filtered=True).reshape(-1)
            probability = filtered_distribution(logits, controls[:, 0], controls[:, 1], controls[:, 3])
            return token, probability

        def execute(logits, controls, acceptance, correction, metadata):
            _, target = distribution(logits[:6], controls[:6])
            proposals, probabilities = [], []
            for index in range(5):
                # Real Markov projections and TP gather are unchanged. Keep
                # five successive sampling calls rather than one C5 sort.
                token, probability = distribution(logits[6 + index:7 + index], controls[6 + index:7 + index])
                proposals.append(token)
                probabilities.append(probability)
            proposed, proposal = torch.cat(proposals), torch.cat(probabilities)
            output, committed, valid = sample_speculative_prefix(target, proposal, proposed, acceptance, correction)
            prefix = verify_control_from_sampled(output, committed, valid, metadata)
            # Next-round probability owner is the first real rejection consumer.
            return *prefix, proposal
        return execute

    report = dict(status='running', vocabulary=vocabulary, credited_e2e_ms=0, real16_qualified=False)
    with torch.inference_mode():
        torch.manual_seed(4211)
        logits = torch.randn(11, vocabulary).to('hpu')
        controls = torch.tensor([[1., .95, .42, -1.]] * 11, device='hpu')
        acceptance = torch.tensor([.1, .3, .5, .7, .9], device='hpu')
        correction = torch.tensor([.37], device='hpu')
        metadata = torch.tensor([1, 6, 5, 1024, 16384, 1048576, 1], dtype=torch.int64, device='hpu')
        functions = [torch.compile(body(candidate), backend='hpu_backend', fullgraph=True, dynamic=False)
                     for candidate in (False, True)]
        weights = (controls, acceptance, correction, metadata)
        for index in range(4):
            logits.add_(torch.linspace(-.03, .03, vocabulary, device='hpu'))
            controls[:, 2].fill_(.13 + .19 * index)
            a, b = [tuple(t.cpu() for t in fn(logits, *weights)) for fn in functions]
            if any(not torch.equal(aa, bb) for aa, bb in zip(a, b, strict=True)):
                raise AssertionError('Single-sort p/q or prefix differs from ordinary HPU sampling')
        replays = [recorder.prepare(fn, [logits], [weights]) for fn in functions]
        try:
            for replay in replays:
                for _ in range(8):
                    replay()
                torch.hpu.synchronize()
            timings = []
            for arm in (0, 1, 0, 1, 0, 1):
                start, end = torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True)
                start.record()
                for _ in range(args.samples):
                    replays[arm]()
                end.record()
                end.synchronize()
                timings.append(dict(arm=arm, device_ms=start.elapsed_time(end) / args.samples))
            report.update(status='passed_exact_component', timings=timings,
                          means_ms=[statistics.fmean(t['device_ms'] for t in timings if t['arm'] == arm)
                                    for arm in (0, 1)])
        finally:
            torch.hpu.synchronize()
            for replay in replays:
                replay.close()
    (root / 'full-probability-chain.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
