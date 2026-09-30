# SPDX-License-Identifier: Apache-2.0
"""Compare additive TP4 activity at a preselected token, preserving all cycles."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics


def coarse(data):
    attention = {'Attention', 'Attention/CSA2', 'CSA2'}
    moe = {'路由专家', '共享专家', 'Router', 'MoE 准备'}
    auxiliary = {'Engram', 'mHC', '入口/尾部/其他张量', '数值边界', '融合表达式待细分', '输出头', '量化'}
    groups = collections.Counter()
    for row in data['overlap_combinations']:
        families = set()
        for name in row['categories']:
            assert name in attention | moe | auxiliary, name
            families.add('Attention / CSA2' if name in attention else 'MoE' if name in moe else 'mHC及其他辅助kernel')
        groups[next(iter(families)) if len(families) == 1 else '跨组重叠（只计一次）'] += row['ms_per_token']
    gaps = collections.Counter()
    other = collections.Counter()
    for row in data['partition']:
        name, value = row['label'], row['ms_per_token']
        if name.startswith('四卡'):
            stage = name.split('：', 1)[1]
            target = ('compiled组内空档' if stage == 'compiled分组入口内' else
                      '结果消费/状态提交空档' if stage == '采样/结果消费与状态提交入口内' else '输入准备/step衔接空档')
            gaps[target] += value
        elif name.startswith('仅'):
            other['仅null描述符' if name.startswith('仅TPC') else '仅DMA活动'] += value
    assert abs(sum(groups.values()) + sum(gaps.values()) + sum(other.values()) - data['period_mean_ms']) < 1e-8
    return dict(compute=dict(groups), gaps=dict(gaps), other=dict(other),
                period_ms=data['period_mean_ms'], gap_ms=sum(gaps.values()))


def distribution(data):
    periods = data['per_cycle']
    values = sorted(row['gap_ms'] for row in periods)
    assert all(abs(sum(row['partition_ms'].values()) - row['period_ms']) < 1e-8 for row in periods)
    at = (len(values) - 1) * .95
    low, high = int(at), min(int(at) + 1, len(values) - 1)
    p95 = values[low] + (values[high] - values[low]) * (at - low)
    return dict(tokens=data['tokens'], cycles=len(values), mean_ms=statistics.mean(values),
                median_ms=statistics.median(values), minimum_ms=min(values), maximum_ms=max(values),
                p95_ms=p95,
                at_most_1p5_ms=sum(value <= 1.5 for value in values), samples=periods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--baseline-all', required=True, type=Path)
    parser.add_argument('--candidate-all', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths = (args.baseline, args.candidate, args.baseline_all, args.candidate_all)
    original, candidate, old_all, new_all = [json.loads(path.read_text()) for path in paths]
    assert original['tokens'] == candidate['tokens'] == [16532]
    assert original.get('unit', 'token') == candidate.get('unit', 'token') == 'token'
    before, after = coarse(original), coarse(candidate)
    old_distribution, new_distribution = distribution(old_all), distribution(new_all)
    result = dict(fixed_position=16532, before=before, after=after,
                  old_distribution=old_distribution, new_distribution=new_distribution,
                  source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  candidate_boundary=candidate.get('boundary_note'),
                  same_clock_scope='Four-rank device union; mutually exclusive classifications within each capture.',
                  limits='Host coincidence does not establish CPU/network causality. No HBM bandwidth counters. '
                         'Trace durations do not replace unprofiled whole-request latency.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# TP4 调度空档对照', '', '固定比较同一逻辑位置 **P16532**，未根据候选延迟重选周期。', '',
             '四卡同一时刻只计一次；以下各项互斥，合计为实际完整周期。', '',
             candidate.get('boundary_note', '连续采样token消费完成之间的完整周期。'), '',
             '| 功能/阶段 | 原 trace ms | 候选 trace ms | 变化 ms |', '|---|---:|---:|---:|']
    for family, names in (
            ('compute', ('Attention / CSA2', 'MoE', 'mHC及其他辅助kernel', '跨组重叠（只计一次）')),
            ('gaps', ('输入准备/step衔接空档', 'compiled组内空档', '结果消费/状态提交空档')),
            ('other', ('仅DMA活动', '仅null描述符'))):
        for name in names:
            a, b = before[family].get(name, 0), after[family].get(name, 0)
            lines.append(f'| {name} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    for name, key in (('完整周期', 'period_ms'), ('无可见设备活动合计', 'gap_ms')):
        a, b = before[key], after[key]
        lines.append(f'| **{name}** | **{a:.6f}** | **{b:.6f}** | **{b-a:+.6f}** |')
    lines += ['', '完整采集窗口的空档分布（不删除异常周期）：', '',
              '| 窗口 | 周期数 | 平均 ms | 中位 ms | p95 ms | 最小–最大 ms | ≤1.5ms 周期 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, d in (('原 trace', old_distribution), ('候选 trace', new_distribution)):
        lines.append(f"| {name} P{d['tokens'][0]}–P{d['tokens'][-1]} | {d['cycles']} | {d['mean_ms']:.6f} | "
                     f"{d['median_ms']:.6f} | {d['p95_ms']:.6f} | {d['minimum_ms']:.6f}–{d['maximum_ms']:.6f} | "
                     f"{d['at_most_1p5_ms']} |")
    lines += ['', '空档按host线程所处阶段定位，不等同于该线程的纯CPU耗时，也不能直接当作NIC通信。'
              '未采集HBM带宽计数器，不能用四卡理论峰值推算此处利用率。'
              '正式性能以不带profiler的完整自然EOS请求为准。', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(fixed_gap_before_ms=before['gap_ms'], fixed_gap_after_ms=after['gap_ms'],
                          all_gap_after_ms=new_distribution['mean_ms']), ensure_ascii=False))


if __name__ == '__main__':
    main()
