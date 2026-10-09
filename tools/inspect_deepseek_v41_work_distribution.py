# SPDX-License-Identifier: Apache-2.0
"""Join cached Gaudi2 static descriptors to ARC work-distribution contexts.

Read-only recipe analysis. Static TID registers do not describe the boxes ARC
actually assigns. Refuse unknown packets, unmatched yields and unsupported
context sizes rather than guessing physical work or SRAM placement.
"""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import struct

from collect_deepseek_v41_trace import recipe_symbols
from inspect_deepseek_v41_recipe_memory import arc_static_descriptors, inspect, register_writes


def tpc_context(payload):
    # gaudi2_arc_eng_packets.h: packed index_space_tensor_t(60), two control
    # words(8), virt_sob_ids_t(12), full_hbm_addr_ctxt_t(8).
    if len(payload) != 88:
        raise ValueError(f'Unsupported Gaudi2 TPC context size {len(payload)}')
    words = struct.unpack('<22I', payload)
    base, grid, box = (list(words[i:i+5]) for i in (0, 5, 10))
    if any(x == 0 for x in grid):
        slices, volumes = [1]*5, [0]
    else:
        if any(x == 0 for x in box):
            raise ValueError('Nonempty TPC context has an empty box')
        slices = [(g+b-1)//b for g, b in zip(grid, box, strict=True)]
        # Multiset only: this does not guess the firmware's physical-core
        # enumeration. Edge boxes perform less work than full boxes.
        volumes = [math.prod(min(b, g-i*b) for g, b, i in zip(grid, box, indices, strict=True))
                   for indices in itertools.product(*(range(n) for n in slices))]
    flags = words[15]
    return dict(base=base, grid=grid, box=box, slices=slices,
                boxes=len(volumes), box_work_histogram=dict(sorted(Counter(volumes).items())),
                largest_box_work=max(volumes), total_work=sum(volumes),
                equal_cost_box_utilization=sum(volumes)/(len(volumes)*max(volumes)) if max(volumes) else 0,
                shuffle_index=flags & 255, enable_two_boxes=bool(flags & (1 << 12)),
                two_box_dim=(flags >> 13) & 7, two_box_tpc_index=flags >> 24,
                control_words=list(words[15:]),
                limitation='Work units are geometry, not cycles, measured ALU utilization or memory bandwidth')


def scheduled_tpc_contexts(data, region, buffers):
    """Contexts are prefetched up to eight slots ahead; preserve DMA order."""
    begin, length = region['offset'], region['bytes']
    if begin+length > len(data):
        raise ValueError('Dynamic ECB exceeds recipe')
    cursor, result, executions = 0, [], 0
    while cursor < length:
        if cursor+4 > length:
            raise ValueError('Truncated dynamic ECB word')
        word = struct.unpack_from('<I', data, begin+cursor)[0]
        opcode, size = word & 15, 4
        if opcode == 1:
            size += 4*(word >> 9)
        elif opcode == 2:
            executions += 1
        elif opcode == 3:
            size = 8
            if cursor+size > length:
                raise ValueError('Truncated dynamic context DMA')
            count, base_index, destination = (word >> 8) & 1023, (word >> 5) & 7, word >> 18
            offset = struct.unpack_from('<I', data, begin+cursor+4)[0]
            if base_index != 2 or count != 88 or destination % 88:
                raise ValueError('Unsupported TPC context DMA binding')
            buffer = buffers[2]
            if offset+count > buffer['bytes']:
                raise ValueError('TPC context DMA exceeds dynamic buffer')
            start = buffer['offset']+offset
            result.append(dict(dynamic_command_offset=cursor, context_slot=destination//88,
                               dynamic_buffer_offset=offset, **tpc_context(data[start:start+count])))
        elif opcode in (6, 7):
            size = 8
        elif opcode != 0:
            raise ValueError(f'Unsupported dynamic ECB opcode {opcode} at {cursor}')
        if cursor+size > length:
            raise ValueError('Dynamic ECB packet exceeds region')
        cursor += size
    if executions != len(result):
        raise ValueError('Scheduled/executed context counts differ')
    return result


def static_tpc_yields(data, region, buffers):
    """A yielded static descriptor corresponds to one ordered logical ROI."""
    begin, length = region['offset'], region['bytes']
    registers, result = {}, []
    for command in arc_static_descriptors(data[begin:begin+length]):
        if command['cpu_index'] not in (0, 254):
            raise ValueError('Unsupported TPC physical-core-specific descriptor')
        if command['buffer_index'] not in (0, 1):
            raise ValueError('Unsupported static TPC buffer binding')
        buffer = buffers[{0: 1, 1: 0}[command['buffer_index']]]
        if command['offset']+command['bytes'] > buffer['bytes']:
            raise ValueError('Static TPC DMA exceeds buffer')
        start = buffer['offset']+command['offset']
        for write in register_writes(data[start:start+command['bytes']]):
            if 'value' in write:
                registers[write['register']] = write['value']
        word = struct.unpack_from('<I', data, begin+command['command_offset'])[0]
        if word & 16:
            if 0xbb18 not in registers:
                raise ValueError('Yielded TPC descriptor lacks a kernel context')
            result.append(dict(context_id=registers[0xbb18] & 65535, command=command))
    return result


def work_distribution(path):
    recipe = inspect(path)
    data, symbols = path.read_bytes(), recipe_symbols(path)
    jobs = [j for j in recipe['arc_jobs'] if j['engine'] == 0]
    if len(jobs) != 1 or jobs[0]['static']['engine_offset']:
        raise ValueError('Expected one Gaudi2 broadcast TPC ARC job')
    job = jobs[0]
    static = static_tpc_yields(data, job['static'], recipe['blob_buffers'])
    dynamic = scheduled_tpc_contexts(data, job['dynamic'], recipe['blob_buffers'])
    if len(static) != len(dynamic):
        raise ValueError('Static ROI yields and dynamic context counts differ')
    by_context = {n['context_id']: n for n in symbols['nodes'] if n['device_type'] == 1}
    joined = [dict(**s, **d, symbol=by_context.get(s['context_id']))
              for s, d in zip(static, dynamic, strict=True)]
    if any(n['enable_two_boxes'] for n in joined):
        raise ValueError('Variable two-box firmware assignment requires a separate work model')
    return dict(recipe=str(path), sha256=recipe['sha256'], recipe_id=recipe['recipe_id'],
                arc_roi_count=len(joined), work_distribution=joined)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = work_distribution(args.recipe)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(recipe_id=result['recipe_id'], arc_roi_count=result['arc_roi_count'])))
