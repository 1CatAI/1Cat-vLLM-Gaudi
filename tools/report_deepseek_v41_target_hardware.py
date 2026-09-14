# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Device-bounded Target accounting; never charge blocked host calls to DMA.

Reuses the saved V4.1 extractor and completion-marker contract. All intervals
are from this acquisition. Function unions can overlap; the exclusive ledger
keeps cross-function overlap as its own row rather than inventing ownership.
"""
import argparse
import collections
import csv
import gzip
import json
from pathlib import Path
import re
import statistics

MEASURED = {
    'TPC': {'TPC_SPU_START_TO_SPU_HALT'},
    'MME': {'MMEH_WB0_MON_TS_BIT1', 'MMEH_WB1_MON_TS_BIT1'},
    'DMA': {'DBG_DMA_TRC_WR_DATA_LAST'}
}


def merge(spans):
    out = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def duration(spans):
    return sum(b - a for a, b in merge(spans))


def partition(named, start, end):
    events = collections.defaultdict(list)
    events[start]
    events[end]
    for name, spans in named.items():
        for a, b in merge(spans):
            a, b = max(a, start), min(b, end)
            if b > a:
                events[a].append((name, 1))
                events[b].append((name, -1))
    result, active = collections.Counter(), collections.Counter()
    bounds = sorted(events)
    for a, b in zip(bounds, bounds[1:]):
        for name, delta in events[a]:
            active[name] += delta
        key = tuple(sorted(name for name, n in active.items() if n))
        result[key] += b - a
    assert abs(sum(result.values()) - (end - start)) < 0.01
    return result


def anchors(root, phases):
    named = {}
    with gzip.open(root / 'host.jsonl.gz', 'rt') as f:
        for line in f:
            ts, dur, pid, tid, cat, name = json.loads(line)
            if name.startswith('v41::clock_anchor::'):
                named[name] = (ts, dur)
    out = {}
    for row in phases['records']:
        anchor = row.get('trace_anchor')
        if not anchor or anchor['name'] not in named:
            continue
        ts, dur = named[anchor['name']]
        # Trace's start/end lie inside these host bounds.
        lo = ts + dur - anchor['host_after_ns'] / 1000
        hi = ts - anchor['host_before_ns'] / 1000
        if lo > hi:
            raise ValueError('Clock anchor bounds disagree')
        c = phases['calibration']
        device_to_host = (c['offset_low_ns'] + c['offset_high_ns']) / 2000
        shift = device_to_host + (lo + hi) / 2
        device = row['device']
        if any(device[name]['status'] != 'complete' for name in ('stage_model_start', 'stage_target_done')):
            continue
        out[row['generation']] = dict(start_us=device['stage_model_start']['device_timestamp_ns'] / 1000 + shift,
                                      end_us=device['stage_target_done']['device_timestamp_ns'] / 1000 + shift,
                                      clock_width_us=hi - lo + (c['offset_high_ns'] - c['offset_low_ns']) / 1000,
                                      host_to_trace_us=(lo + hi) / 2,
                                      record=row)
    return out


def classify(node, contract=None):
    name, kernel, engine = node.get('node', ''), node['kernel'], node['engine']
    text = (name + ' ' + kernel).lower()
    if engine == 'DMA':
        if 'pdma_tx_commands' in text:
            return '通信与搬运', '命令页 H2D'
        if 'transpose' in text:
            return '通信与搬运', '硬件转置'
        return '通信与搬运', 'DMA（用途见源节点）'
    ins = (contract or {}).get('inputs', [])
    outs = (contract or {}).get('outputs', [])
    shapes = [t.get('shape') for t in ins]
    origin = (contract or {}).get('fused_original_nodes', [])
    # In this measured <=512 search bucket the CSA2 candidate selection does
    # not sort scores. Bitonic/Top-k device nodes in Target are MoE routing.
    if any(x in text for x in ('bitonic', 'topk', 'merge_sort')):
        return 'Router', 'Top-6：排序、索引及权重选择'
    if engine != 'MME' and any(len(s or []) == 2 and s[0] >= 64640 and s[1] == 5120 for s in shapes):
        return '输入与尾部', 'Embedding 分片行读取'
    if kernel.startswith('fused_kernel') and origin:
        roles = set()
        for source in origin:
            sn = source['name']
            if source.get('semantic_group'):
                roles.add(source['semantic_group'])
            elif '/attention/' in sn:
                roles.add('CSA2/KV/Compressor')
            elif '/moe/' in sn:
                roles.add('Router' if any(384 in (t.get('shape') or []) for t in ins + outs) else '专家解码与计算')
            elif '/layers/' in sn:
                roles.add('mHC/Norm')
            else:
                roles.add('输入与尾部')
        ops = list(
            dict.fromkeys(n['op'] for n in origin
                          if n['op'] not in ('Reshape', 'ExpandDims', 'Slice', 'Squeeze', 'StaticReshape')))
        if len(roles) == 1:
            return roles.pop(), '融合：' + ' → '.join(ops)
        return '跨模块融合', '融合：' + ' → '.join(ops)
    if engine == 'MME' and [25600, 6144] in shapes:
        return 'Engram', 'Wkv 6144→25600 投影'
    if engine == 'MME' and [24, 20480] in shapes:
        return 'mHC/Norm', '20480→24 控制投影（FP32）'
    if 'mxfp4' in text:
        if 'dequant' in kernel:
            stage = 'W13' if any(5120 in (t.get('shape') or []) and 2304 in (t.get('shape') or [])
                                 for t in outs) else 'W2'
            return '专家解码与计算', stage + ' MXFP4→BF16 专家解码'
        if engine == 'MME':
            stage = 'W13' if any(2304 in (t.get('shape') or []) for t in ins + outs) else 'W2'
            return '专家解码与计算', stage + ' 专家矩阵'
        return '专家解码与计算', '专家激活/路由归约'
    if '/moe/' in name:
        if engine == 'MME':
            if any(t.get('dtype') in ('float32', 'float') and 384 in (t.get('shape') or []) for t in ins + outs):
                return 'Router', '5120→384 路由投影'
            if 'linear_fwd_f32' in name:
                return 'Router', '路由投影（源调用序号）'
            return '专家解码与计算', '共享专家投影'
        if any(x in text for x in ('topk', 'bitonic', 'gather', 'log1p', 'softplus', 'sqrt', 'exp', 'route')):
            return 'Router', '打分/Top-6/路由权重'
        return '专家解码与计算', 'MoE 辅助（融合归属待图关联）'
    if '/attention/' in name:
        if engine == 'MME':
            if 'bmm/' in name or 'batchgemm' in text:
                return 'Attention 投影', 'MLA wo_a 输出 BMM'
            projection = next((label for shape, label in (
                ([1280, 5120], 'Q 低秩 wq_a 5120→1280'),
                ([16384, 1280], 'Q 展开 wq_b 1280→16384'),
                ([128, 512], 'index K 512→128'),
                ([5120, 4096], 'wo_b 输出 4096→5120'),
                ([512, 5120], 'KV/Compressor 5120→512（保留节点区别）'),
            ) if shape in shapes), 'Attention 线性投影')
            return 'Attention 投影', projection
        return 'CSA2/KV/Compressor', 'Attention 内非矩阵运算'
    if any(x in text
           for x in ('paged_attention', 'sparse_attn', 'pack_fp4', 'pack_swa', 'rope', 'kv_cache', 'compressor')):
        return 'CSA2/KV/Compressor', '稀疏 attention / KV / 索引 / RoPE'
    if 'engram' in text:
        return 'Engram', '行消费 / Wkv / 状态融合'
    if any(x in text for x in ('sinkhorn', 'mhc', 'hc_pre', 'hc_post', 'pre_mix', 'rms_norm')):
        return 'mHC/Norm', 'mixing / statistics / Sinkhorn / norm'
    if re.search(r'/layers/\d+/(linear|einsum|bmm)(?:_|/)', name) and engine == 'MME':
        return 'mHC/Norm', 'mHC 控制 / 通道混合矩阵'
    if ('/layers/' in name and '/moe/' not in name and '/attention/' not in name and any(
        (t.get('shape') or [])[-1:] in ([24], [4]) for t in ins + outs)):
        # Scope-derived attribution is intentionally limited to explicit
        # mHC control shapes; Engram also lives at the decoder-layer scope.
        return 'mHC/Norm', '控制权重 / 4×4 通道统计'
    if 'embedding' in text or 'norm/' in text or 'target_state' in text:
        return '输入与尾部', 'Embedding / final norm / target states'
    return '未完整归因的计算', '保留实测；待源图关联'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--analysis', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--discard', type=int, default=2)
    p.add_argument('--take', type=int, default=3)
    args = p.parse_args()
    run = args.run.resolve()
    analysis = args.analysis or run / 'trace-analysis'
    out = args.output or run / 'target-hardware-breakdown'
    out.mkdir(exist_ok=False)
    inventories = {rank: json.loads((analysis / f'rank{rank}/inventory.json').read_text()) for rank in range(4)}
    phases = {rank: json.loads((run / f'verify-phases/rank{rank}-verify-phases.json').read_text()) for rank in range(4)}
    windows = {rank: anchors(analysis / f'rank{rank}', phases[rank]) for rank in range(4)}
    common = sorted(set.intersection(*(set(w) for w in windows.values())))
    selected = common[args.discard:args.discard + args.take]
    if len(selected) != args.take:
        raise ValueError(f'Insufficient paired C6 windows: {common}')
    summary = {
        'scope': 'C6 Target stage_model_start → stage_target_done; excludes head/prefix/draft/commit',
        'units': 'ms',
        'generations': selected,
        'discarded': common[:args.discard],
        'acquisition': str(run),
        'ranks': [],
        'pairs': []
    }
    all_detail = []
    for rank in range(4):
        inv = inventories[rank]
        nodes = inv['nodes']
        hw = inv['hw_event_names']
        contracts_path = analysis / f'rank{rank}/enriched-contracts.json'
        if not contracts_path.exists():
            contracts_path = analysis / f'rank{rank}/node-contracts.json'
        cindex = collections.defaultdict(list)
        if contracts_path.exists():
            for c in json.loads(contracts_path.read_text()):
                cindex[(str(c['recipe_id']), c['symbol']['node'])].append(c)
        contracts = {}
        for ni, node in enumerate(nodes):
            cs = cindex[(node['recipe'].split(':')[0], node['node'])]
            matches = [c for c in cs if c['matched']]
            if matches and all(
                    c.get('inputs') == matches[0].get('inputs') and c.get('outputs') == matches[0].get('outputs')
                    for c in matches):
                contracts[ni] = matches[0]
        kinds = {ni: classify(node, contracts.get(ni)) for ni, node in enumerate(nodes)}
        lane_names = {
            str(m['tid']): m['args']['name']
            for m in inv['metadata'] if m.get('name') == 'thread_name' and m.get('pid') == 0
        }
        samples = {
            g:
            dict(groups=collections.defaultdict(list),
                 engines=collections.defaultdict(list),
                 lanes=collections.defaultdict(list),
                 details=collections.defaultdict(list))
            for g in selected
        }
        with gzip.open(analysis / f'rank{rank}/hardware.jsonl.gz', 'rt') as f:
            for line in f:
                ts, dur, lane, ni, hi = json.loads(line)
                engine = nodes[ni]['engine']
                if hw[hi] not in MEASURED.get(engine, set()):
                    continue
                for gen, s in samples.items():
                    w = windows[rank][gen]
                    a, b = max(ts, w['start_us']), min(ts + dur, w['end_us'])
                    if b <= a:
                        continue
                    s['groups'][kinds[ni][0]].append((a, b))
                    s['engines'][engine].append((a, b))
                    s['lanes'][(engine, lane)].append((a, b))
                    s['details'][ni].append((a, b, lane, hi, ts, dur))
        total_window = sum(windows[rank][g]['end_us'] - windows[rank][g]['start_us'] for g in selected)
        groups = collections.Counter()
        exclusive = collections.Counter()
        engines = collections.Counter()
        lane_times = collections.Counter()
        engine_partition = collections.Counter()
        details = collections.defaultdict(list)
        for g, s in samples.items():
            w = windows[rank][g]
            for name, spans in s['groups'].items():
                groups[name] += duration(spans)
            for key, value in partition(s['groups'], w['start_us'], w['end_us']).items():
                exclusive[key[0] if len(key) == 1 else '跨功能组重叠' if key else '硬件活动未覆盖区间'] += value
            for key, value in partition(s['engines'], w['start_us'], w['end_us']).items():
                engine_partition[' + '.join(key) or 'uncovered'] += value
            for name, spans in s['engines'].items():
                engines[name] += duration(spans)
            for lane, spans in s['lanes'].items():
                lane_times[lane] += duration(spans)
            for ni, events in s['details'].items():
                details[ni].append((g, events))
        for ni, observations in details.items():
            node = nodes[ni]
            contract = contracts.get(ni)
            activity = sum(duration([(e[0], e[1]) for e in events]) for g, events in observations)
            detail = dict(rank=rank,
                          group=kinds[ni][0],
                          purpose=kinds[ni][1],
                          **node,
                          activity_ms_per_C6=activity / len(selected) / 1000,
                          share_pct=activity / total_window * 100,
                          lane_packets=sum(len(events) for g, events in observations),
                          physical_calls=None,
                          mean_call_ms=None,
                          physical_call_note='Not inferred from lane packet count',
                          contract=contract,
                          observations=[dict(generation=g, events=events) for g, events in observations])
            all_detail.append(detail)

        def rows(counter, total_window=total_window):
            return [
                dict(name=name, ms_per_C6=value / len(selected) / 1000, share_pct=value / total_window * 100)
                for name, value in counter.most_common()
            ]

        result = dict(rank=rank,
                      stage_window_ms=total_window / len(selected) / 1000,
                      max_clock_uncertainty_ms=max(windows[rank][g]['clock_width_us'] for g in selected) / 1000,
                      groups=rows(groups),
                      exclusive=rows(exclusive),
                      engines=rows(engines),
                      engine_partition=rows(engine_partition),
                      lanes=[
                          dict(engine=e,
                               lane=lane,
                               name=lane_names.get(lane, lane),
                               active_ms_per_C6=v / len(selected) / 1000,
                               duty_pct=v / total_window * 100) for (e, lane), v in sorted(lane_times.items())
                      ],
                      kernel_rows=len(details),
                      matched_contract_rows=sum(ni in contracts for ni in details))
        summary['ranks'].append(result)
    for tp in range(2):
        paired = []
        for g in selected:
            a, b = windows[tp][g], windows[tp + 2][g]
            paired.append(
                dict(generation=g,
                     pp0_ms=(a['end_us'] - a['start_us']) / 1000,
                     handoff_ms=(b['start_us'] - a['end_us']) / 1000,
                     pp1_ms=(b['end_us'] - b['start_us']) / 1000,
                     total_ms=(b['end_us'] - a['start_us']) / 1000))
        summary['pairs'].append(
            dict(tp=tp,
                 samples=paired,
                 mean={
                     key: statistics.mean(r[key] for r in paired)
                     for key in ('pp0_ms', 'handoff_ms', 'pp1_ms', 'total_ms')
                 }))
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with gzip.open(out / 'kernel-details.json.gz', 'wt') as f:
        json.dump(all_detail, f, ensure_ascii=False)
    with (out / 'kernel-details.csv').open('w') as f:
        fields = [
            'rank', 'group', 'purpose', 'engine', 'kernel', 'node', 'recipe', 'reported_dtype', 'activity_ms_per_C6',
            'share_pct', 'lane_packets', 'physical_calls', 'mean_call_ms'
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_detail)
    (out / 'selected-windows.json'
     ).write_text(json.dumps({rank: {
         g: windows[rank][g]
         for g in selected
     }
                              for rank in range(4)}, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
