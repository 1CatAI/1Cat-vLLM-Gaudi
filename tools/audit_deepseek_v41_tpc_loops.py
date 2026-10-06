# SPDX-License-Identifier: Apache-2.0
"""Reject compiled TPC empty self-loops before opening a device.

A hardware/MMIO polling intrinsic can be hoisted out of a C loop. Simulation
may miss this when its first read is already complete. This audit only detects
an unconditional backedge whose entire body is NOP/jump; it does not prove the
correctness of other polling or synchronization protocols.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def empty_self_loops(assembly):
    if 'TPC architecture: gaudi2' not in assembly:
        raise ValueError('Require genuine Gaudi2 TPC disassembly')
    labels, pending, instructions = {}, [], []
    for line in assembly.splitlines():
        label = re.search(r'(\.LBB[\w_]+):\s*$', line)
        if label:
            pending.append(label[1])
        record = re.match(r'\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)+\s*(.*?)\s*$', line)
        if not record:
            continue
        address = int(record[1], 16)
        for name in pending:
            labels[name] = address
        pending.clear()
        operations = record[2].split('//', 1)[0].strip()
        instructions.append((address, operations))
    failures = []
    for address, ops in instructions:
        jump = re.search(r'\bjmpr\s+(\.LBB[\w_]+)\s*(?:;|$)', ops)
        if not jump or jump[1] not in labels or labels[jump[1]] > address:
            continue
        first = labels[jump[1]]
        body = [text for at, text in instructions if first <= at <= address]
        if all(not re.sub(r'\b(?:nop|jmpr\s+\.LBB[\w_]+)\b|[;\s]', '', text) for text in body):
            failures.append(dict(target=jump[1], start=hex(first), branch=hex(address), body=body))
    return failures


def audit_object(path):
    result = subprocess.run(['tpc-llvm-objdump', '--arch-name=tpc', '--mcpu=gaudi2', '-d',
                             str(path)],
                            check=True,
                            text=True,
                            capture_output=True)
    return dict(object=str(path), empty_self_loops=empty_self_loops(result.stdout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('objects', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [audit_object(path) for path in args.objects]
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    if any(row['empty_self_loops'] for row in results):
        raise SystemExit('Compiled TPC empty self-loop: artifact rejected before device run')


if __name__ == '__main__':
    main()
