# SPDX-License-Identifier: Apache-2.0
"""Validate Gaudi2 U8 shuffle against saved SDK simulator lane probes.

This is an instruction-semantics check, not a performance benchmark. In each
64-byte dual group the 32-lane mux uses the direction at the selected position
to choose its source group. A flat 64-entry CPU lookup is not an oracle.
"""
import argparse
import json
from pathlib import Path


def shuffle(values, directions, income):
    out = list(income)
    for lane, direction in enumerate(directions):
        if not direction & 128:
            continue
        base = lane // 64 * 64
        element = direction & 31
        mux_lane = base + (lane % 64 // 32) * 32 + element
        source = base + (directions[mux_lane] & 32) + element
        out[lane] = values[source]
    return out


def check_probe(path):
    raw = path.read_bytes()
    if len(raw) != 128 * 256:
        raise ValueError('Expected 128 rows of 256 simulator output bytes')
    mismatches = 0
    for offset in range(128):
        directions = [((lane + offset) & 255) | 128 for lane in range(256)]
        expected = shuffle(list(range(256)), directions, directions)
        mismatches += sum(a != b for a, b in zip(raw[offset * 256:(offset + 1) * 256], expected))
    return mismatches


def dictionary_check():
    """Expose the remaining mux error *after* explicit nibble masking."""
    table = [(lane // 64 * 64 + lane % 64) for lane in range(256)]
    cases, wrong = 0, 0
    example = None
    for pattern in range(5):
        for upper in range(16):
            raw = [((lane * (pattern * 2 + 1) + pattern) & 15) | upper << 4 for lane in range(256)]
            classes = [(lane + pattern) % 4 for lane in range(256)]
            directions = [128 | ((value & 15) << 2) | cls for value, cls in zip(raw, classes)]
            actual = shuffle(table, directions, [0] * 256)
            expected = [table[lane // 64 * 64 + (directions[lane] & 63)] for lane in range(256)]
            for lane, (a, b) in enumerate(zip(actual, expected)):
                cases += 1
                if a != b:
                    wrong += 1
                    if example is None:
                        example = dict(lane=lane, raw=raw[lane], direction=directions[lane],
                                       actual=a, flat_lookup=b)
    return dict(lanes_checked=cases, flat_dictionary_mismatches=wrong, counterexample=example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator-probe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    errors = check_probe(args.simulator_probe)
    result = dict(simulator_lane_comparisons=32768, simulator_model_mismatches=errors,
                  masked_dictionary=dictionary_check(), two_instruction_decoder_qualified=False,
                  conclusion='Masking removes upper-nibble contamination but does not fix the 64-entry dictionary mux.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if errors:
        raise SystemExit('Shuffle model disagrees with the SDK simulator')


if __name__ == '__main__':
    main()
