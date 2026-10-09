# SPDX-License-Identifier: Apache-2.0
"""Read cached Synapse recipe bindings offline; do not infer SRAM from bundles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

from collect_deepseek_v41_trace import recipe_symbols


class Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def read(self, fmt):
        size = struct.calcsize('<'+fmt)
        if self.offset+size > len(self.data):
            raise ValueError('Recipe field exceeds binary')
        values = struct.unpack_from('<'+fmt, self.data, self.offset)
        self.offset += size
        return values[0] if len(values) == 1 else values

    def skip(self, size):
        if not 0 <= size <= len(self.data)-self.offset:
            raise ValueError('Recipe region exceeds binary')
        self.offset += size

    def name(self):
        length = self.read('I')
        if not 0 < length < 16384:
            raise ValueError('Invalid tensor name length')
        value = self.data[self.offset:self.offset+length]
        self.skip(length)
        if value[-1:] != b'\0':
            raise ValueError('Tensor name lacks terminator')
        return value[:-1].decode()


def persistent(reader):
    result = dict(name=reader.name())
    result['section'], result['offset'], result['bytes'] = reader.read('HQQ')
    dtype, _, _, dimensions = reader.read('IddI')
    size_words = reader.read('25I')
    # ABI1.26's serializer copies 25 uint32 words from a TSize[25]
    # (uint64) array. Decode complete low/high pairs, never report the high
    # word as another dimension. Larger ranks have truncated geometry.
    if dimensions > len(size_words)//2:
        raise ValueError('Serialized ABI1.26 truncates this tensor geometry')
    sizes = [size_words[2*i]+(size_words[2*i+1] << 32) for i in range(dimensions)]
    reader.skip(25)
    _, is_input, is_external, execution_order, _, _ = reader.read('IBBIBB')
    reader.skip(reader.read('I'))
    reader.skip(4*reader.read('Q'))
    result.update(dtype=dtype, sizes=sizes, is_input=bool(is_input),
                  is_external=bool(is_external), external_execution_order=execution_order)
    return result


def register_writes(data):
    """Decode Gaudi2 QMAN register packets, refusing unknown instructions."""
    cursor, writes = 0, []
    while cursor < len(data):
        low, control = struct.unpack_from('<II', data, cursor)
        opcode = (control >> 24) & 31
        if opcode == 1:
            writes.append(dict(register=(control >> 8) & 65535, value=low, packet_offset=cursor))
            size = 8
        elif opcode == 2:
            count = low & 65535
            start = (control >> 8) & 65535
            values = struct.unpack_from('<'+str(count*2)+'I', data, cursor+8)
            writes.extend(dict(register=start+i*4, value=v, packet_offset=cursor) for i, v in enumerate(values))
            size = 8+count*8
        elif opcode in (18, 19):
            start = ((control >> 10) & 16383)*4
            base = (control >> 5) & 15
            if opcode == 18:
                offset, relative, enable = low, True, 3
                size = 8
            else:
                offset = struct.unpack_from('<Q', data, cursor+8)[0]
                relative, enable = bool((control >> 9) & 1), low & 3
                size = 16
            if relative:
                writes.append(dict(register=start, relative_base_index=base, address_offset=offset,
                                   enabled_words=enable, packet_offset=cursor))
            else:
                for word in range(2):
                    if enable & (1 << word):
                        writes.append(dict(register=start+word*4, value=(offset >> (word*32)) & 0xffffffff,
                                           packet_offset=cursor))
        elif opcode in (4, 8, 10, 11, 12, 13):
            size = 8
        elif opcode in (3, 5, 7, 14, 15, 16):
            size = 16
        elif opcode == 9:
            size = 24
        else:
            raise ValueError(f'Unknown QMAN opcode {opcode} at {cursor}')
        cursor += size
    if cursor != len(data):
        raise ValueError('QMAN packet sizes do not exhaust blob')
    return writes


def arc_static_descriptors(data):
    """Read static ECB commands using gaudi2_arc_eng_packets.h bit fields."""
    cursor, result = 0, []
    while cursor < len(data):
        word = struct.unpack_from('<I', data, cursor)[0]
        opcode = word & 15
        if opcode == 0:
            size = 4
        elif opcode == 1:
            size = 4+4*(word >> 9)
        elif opcode == 5:
            result.append(dict(command_offset=cursor, cpu_index=(word >> 8) & 255,
                               bytes=(word >> 16) & 8191, buffer_index=(word >> 29) & 7,
                               offset=struct.unpack_from('<I', data, cursor+4)[0]))
            size = 8
        else:
            raise ValueError(f'Unknown static ECB opcode {opcode} at {cursor}')
        if size > len(data)-cursor:
            raise ValueError('Static ECB packet exceeds its engine region')
        cursor += size
    return result


def address_binding(registers, start):
    low, high = registers.get(start), registers.get(start+4)
    if low is None:
        return None
    if 'value' in low and high and 'value' in high:
        address = low['value']+(high['value'] << 32)
        return dict(address=hex(address),
                    proven_sram=0x1000fffffd000000 <= address < 0x1001000000000000)
    return dict(binding=low, proven_sram=False)


def replay_static_registers(data, jobs, buffers, symbols):
    """Compose masked register updates in each engine's actual static ECB order.

    Addresses below physical SRAM base can be ROI-biased, and are left unknown.
    Relative runtime bases are symbolic; never retain a stale absolute high word.
    """
    result, rejected = [], []
    base_to_buffer = {0: 1, 1: 0, 2: 2}
    for job in jobs:
        if job['engine'] not in (0, 1):
            continue
        region = job['static']
        chunk = region['engine_offset'] or region['bytes']
        for engine, begin in enumerate(range(0, region['bytes'], chunk)):
            registers = {}
            try:
                commands = arc_static_descriptors(data[region['offset']+begin:
                                                      region['offset']+min(begin+chunk, region['bytes'])])
                for command in commands:
                    if command['cpu_index'] not in (engine, 254):
                        continue
                    if command['buffer_index'] == 7:
                        continue
                    buffer = buffers[base_to_buffer[command['buffer_index']]]
                    if command['offset']+command['bytes'] > buffer['bytes']:
                        raise ValueError('ECB refers outside its recipe buffer')
                    start = buffer['offset']+command['offset']
                    writes = register_writes(data[start:start+command['bytes']])
                    for write in writes:
                        reg = write['register']
                        if 'relative_base_index' in write:
                            for i in range(2):
                                if write['enabled_words'] & (1 << i):
                                    registers[reg+4*i] = write
                        else:
                            registers[reg] = write
                    record = dict(engine_type=job['engine'], engine_index=engine, command=command)
                    if job['engine'] == 0 and any(w['register'] == 0xbb18 for w in writes):
                        context = registers[0xbb18]['value'] & 65535
                        record.update(context_id=context,
                                      symbols=[n for n in symbols['nodes'] if n['device_type'] == 1
                                               and n['context_id'] == context],
                                      tensors=[dict(slot=slot, **binding) for slot in range(16)
                                               if (binding := address_binding(registers, 0xb5dc+80*slot))])
                        result.append(record)
                    elif job['engine'] == 1 and any(0xb008 <= w['register'] < 0xb028 for w in writes):
                        record['tensors'] = [dict(operand=name, **binding) for name, reg in
                                             (('Cout1', 0xb008), ('Cout0', 0xb010),
                                              ('A', 0xb018), ('B', 0xb020))
                                             if (binding := address_binding(registers, reg))]
                        result.append(record)
            except (ValueError, KeyError, struct.error) as error:
                rejected.append(dict(engine_type=job['engine'], engine_index=engine, error=str(error)))
    return result, rejected


def inspect(path, *, allow_missing_symbols=False):
    data = path.read_bytes()
    r = Reader(data)
    version = r.read('II')
    if version != (1, 26):
        raise ValueError(f'Unsupported serializer ABI {version}; do not guess offsets')
    buffers = []
    for _ in range(3):
        size = r.read('Q')
        buffers.append(dict(offset=r.offset, bytes=size))
        r.skip(size)
    blobs = [dict(zip(('type', 'bytes', 'offset'), r.read('BIQ'), strict=True)) for _ in range(r.read('Q'))]
    programs = r.read('I')
    for _ in range(programs):
        r.skip(8*r.read('I'))
    for _ in range(2):
        r.skip(8*r.read('I'))
    jobs = []
    for _ in range(r.read('I')):
        job = dict(engine=r.read('B'), engines_filter=r.read('I'))
        for _ in range(2):
            size, engine_offset = r.read('II')
            job['static' if 'static' not in job else 'dynamic'] = dict(
                bytes=size, engine_offset=engine_offset, offset=r.offset)
            r.skip(size)
        jobs.append(job)
    persistent_tensors = [persistent(r) for _ in range(r.read('I'))]
    h2di = r.read('I')
    views = [persistent(r) for _ in range(r.read('Q'))]
    r.skip(r.read('Q'))
    for _ in range(r.read('I')):
        size, _, present = r.read('QHB')
        if present:
            r.skip(size)
    r.skip(26*r.read('I'))
    count, active = r.read('II')
    patches = []
    for _ in range(count):
        kind, blob, word = r.read('BII')
        payload = dict(tensor_db_index=r.read('I')) if kind == 3 else dict(zip(
            ('effective_address', 'section'), r.read('QH'), strict=True))
        patches.append(dict(type=kind, blob=blob, word=word, node_execution=r.read('I'), **payload))
    for _ in range(r.read('I')+1):
        r.skip(1)
        r.skip(4*r.read('I'))
    for _ in range(r.read('I')):
        r.skip(2)
        r.skip(4*r.read('I'))
    nodes = r.read('I')
    r.skip(nodes*(4+4*programs))
    sections = r.read('I')
    workspaces = r.read(str(r.read('I'))+'Q')
    debug_verified = True
    try:
        symbols = recipe_symbols(path)
    except ValueError as error:
        if not allow_missing_symbols or 'candidates=0' not in str(error):
            raise
        symbols = dict(recipe_id=None, nodes=[])
        debug_verified = False
    if debug_verified and r.offset != symbols['offset']:
        raise ValueError('Serializer parser did not reach the independently verified debug section')
    for patch in patches:
        if patch['blob'] >= len(blobs) or patch['word']*4+4 > blobs[patch['blob']]['bytes']:
            raise ValueError('Patch point refers outside its blob')
    descriptors, rejected = [], []
    for index, blob in enumerate(blobs):
        if blob['type'] != 2:
            continue
        begin = buffers[0]['offset']+blob['offset']
        try:
            writes = register_writes(data[begin:begin+blob['bytes']])
        except (ValueError, struct.error) as error:
            rejected.append(dict(blob=index, error=str(error)))
            continue
        registers = {w['register']: w for w in writes}
        # ARC programs the QM shadow descriptor, rather than the live
        # kernel descriptor. gaudi2_blocks.h: QM tensor0=0x15dc,
        # QM non-tensor=0x1ae4; the queue window adds0xa000.
        context_register, tensor_base = 0xbb18, 0xb5dc
        if context_register not in registers or 'value' not in registers[context_register]:
            continue
        context = registers[context_register]['value'] & 0xffff
        matches = [n for n in symbols['nodes'] if n['device_type'] == 1 and n['context_id'] == context]
        tensors = []
        for slot in range(16):
            low, high = registers.get(tensor_base+slot*80), registers.get(tensor_base+4+slot*80)
            if low is None:
                continue
            if 'value' in low and high and 'value' in high:
                address = low['value']+(high['value'] << 32)
                tensors.append(dict(slot=slot, address=hex(address),
                    proven_sram=0x1000fffffd000000 <= address < 0x1001000000000000))
            else:
                tensors.append(dict(slot=slot, binding=low, proven_sram=False))
        descriptors.append(dict(blob=index, context_id=context, symbols=matches, tensors=tensors))
    ordered, unparsed_arcs = replay_static_registers(data, jobs, buffers, symbols)
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), version=list(version),
                recipe_id=symbols['recipe_id'], blob_buffers=buffers, blobs=blobs, patches=patches,
                persistent_tensors=persistent_tensors, permute_views=views, h2di_tensors=h2di,
                active_patch_points=active, section_count=sections, workspace_sizes=list(workspaces),
                node_count=nodes, patch_sections=dict(Counter(p.get('section') for p in patches)),
                debug_offset_verified=debug_verified, trailing_bytes=len(data)-r.offset,
                tpc_static_descriptors=descriptors, unparsed_execution_blobs=rejected,
                arc_jobs=jobs, ordered_static_descriptors=ordered, unparsed_arcs=unparsed_arcs,
                placement_scope=('Absolute SRAM bindings are proven; '
                                 'ROI-biased and runtime-relative addresses remain unknown'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-missing-symbols', action='store_true',
                        help='Decode unprofiled structural fields and bindings; symbols and tail unverified')
    args = parser.parse_args()
    result = inspect(args.recipe, allow_missing_symbols=args.allow_missing_symbols)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('recipe_id', 'workspace_sizes', 'patch_sections',
                                           'debug_offset_verified', 'placement_scope')}))
