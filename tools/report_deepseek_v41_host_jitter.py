# SPDX-License-Identifier: Apache-2.0
"""Summarize saved token arrivals and correlate spikes with one-second host samples."""
import argparse
import json
from pathlib import Path
import statistics


def distribution(samples):
    if not samples:
        raise ValueError('No individual token intervals')
    ordered = sorted(samples)

    def percentile(fraction):
        point = (len(ordered) - 1) * fraction
        low = int(point)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (point - low)

    median = percentile(.5)
    excess = statistics.mean(max(0, sample - median) for sample in samples)
    return dict(count=len(samples), mean_ms=statistics.mean(samples), median_ms=median,
                p25_ms=percentile(.25), p90_ms=percentile(.9), p99_ms=percentile(.99),
                over_15ms_count=sum(sample > 15 for sample in samples),
                over_15ms_percent=100 * sum(sample > 15 for sample in samples) / len(samples),
                positive_excess_above_median_ms_per_token=excess,
                excess_at_most_point2_ms=excess <= .2)


def report(result, host):
    saved = result.get('client_inter_token_latency', {})
    if not saved.get('valid'):
        raise ValueError('Coalesced arrivals cannot establish individual-token jitter')
    samples = saved['samples_ms']
    output = dict(client_itl=distribution(samples),
                  excess_definition='mean(max(ITL - median(ITL), 0)), including all individual intervals',
                  metric_scope='Client arrival includes engine, IPC, HTTP transport and client scheduling',
                  request_start_ns=result['request_start_ns'], spikes=[])
    # Keep official engine TPOT separate from client-arrival statistics.
    for key in ('engine', 'decode_ms_per_token', 'status', 'tokens'):
        if key in result:
            output[key] = result[key]
    records = [item for item in host if item.get('type') != 'metadata']
    events = result.get('sse_events', [])
    if events and len(events) != len(samples) + 1:
        raise ValueError('Token event count does not match individual intervals')
    if events:
        first = result['request_start_ns'] + round(events[0]['arrival_s'] * 1e9)
        last = result['request_start_ns'] + round(events[-1]['arrival_s'] * 1e9)
        records = [item for item in records if item['wall_ns'] >= first and
                   item['wall_ns'] - round(item['interval_s'] * 1e9) <= last]
        output['host_scope'] = dict(first_token_ns=first, last_token_ns=last,
                                    selection='one-second bins intersecting first-to-last token interval')
    for index, value in enumerate(samples):
        if value <= 15:
            continue
        spike = dict(interval_index=index, itl_ms=value)
        if events:
            left = result['request_start_ns'] + round(events[index]['arrival_s'] * 1e9)
            right = result['request_start_ns'] + round(events[index + 1]['arrival_s'] * 1e9)
            spike.update(start_ns=left, end_ns=right,
                host_sample_indices=[i for i, item in enumerate(records)
                                     if item['wall_ns'] >= left and
                                     item['wall_ns'] - round(item['interval_s'] * 1e9) <= right])
        output['spikes'].append(spike)
    if records:
        pressures = [float(dict(part.split('=') for part in r['cpu_pressure'].splitlines()[0].split()[1:])['avg10'])
                     for r in records]
        output['host'] = dict(samples=len(records), load1_min=min(r['load1'] for r in records),
                             load1_max=max(r['load1'] for r in records),
                             cpu_some_avg10_max_percent=max(pressures),
                             all_recorded_cpu_some_avg10_below_1=all(p < 1 for p in pressures),
                             load1_is_not_an_acceptance_gate=True,
                             one_second_sampling_cannot_prove_per_token_causality=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--host-samples', type=Path)
    args = parser.parse_args()
    host = [json.loads(line) for line in args.host_samples.read_text().splitlines()] if args.host_samples else []
    result = report(json.loads(args.result.read_text()), host)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'spikes'}))


if __name__ == '__main__':
    main()
