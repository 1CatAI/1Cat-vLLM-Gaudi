# SPDX-License-Identifier: Apache-2.0
"""Render an additive, phase-localized ledger for saved Target hardware gaps."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from report_deepseek_v41_target_gaps import host_class, intersections
from report_deepseek_v41_target_hardware import merge

LABELS = {
    'PP0 input preparation before native replay': 'PP0 重放提交前：embedding、Engram staging、输入绑定',
    'PP0 after native replay submission': 'PP0 重放已提交后的未覆盖区间',
    'PP1 snapshot and native capture before instantiate barrier': 'PP1 状态快照和原生逐段捕获',
    'PP1 instantiate stream barrier': 'PP1 实例化前的 stream synchronize',
    'PP1 native plan finalization before state restore': 'PP1 同步返回后→状态恢复前：计划组装、绑定和 pipeline 收尾',
    'PP1 state restore and final replay enqueue': 'PP1 状态恢复和最终重放提交',
    'PP1 queued work after final replay enqueue': 'PP1 最终重放已提交后的未覆盖区间',
    'PP1 invalidation or inter-group boundary': 'PP1 失效清理、输入重绑和组间边界',
    'PP1 five group plan preparation and ordinary execution': 'PP1 五个编译分组：计划重新准备和普通逐段执行',
    'PP1 work after last group wrapper': 'PP1 最后分组返回后的边界',
}


def analyze_phase_host(root, rank, phases):
    named = collections.defaultdict(list)
    for row in phases:
        named[row['phase']].extend(row['gaps'])
    named = {key: merge(value) for key, value in named.items()}
    ends = {key: [b for a, b in value] for key, value in named.items()}
    hits = collections.defaultdict(lambda: collections.defaultdict(list))
    with gzip.open(root / f'trace-analysis/rank{rank}/host.jsonl.gz', 'rt') as f:
        for line in f:
            ts, dur, pid, tid, cat, name = json.loads(line)
            kind = host_class(cat, name)
            if not kind:
                continue
            for phase, spans in named.items():
                clipped = list(intersections(ts, ts+dur, spans, ends[phase]))
                hits[phase][kind].extend(clipped)
    results = {}
    count = len({row['generation'] for row in phases})
    for phase, spans in named.items():
        boundaries = collections.defaultdict(list)
        for a, b in spans:
            boundaries[a].append(('gap', 1)); boundaries[b].append(('gap', -1))
        unions = {}
        for kind, intervals in hits[phase].items():
            merged = merge(intervals)
            unions[kind] = sum(b-a for a,b in merged)/count/1000
            for a,b in merged:
                boundaries[a].append((kind, 1)); boundaries[b].append((kind, -1))
        active, ledger = collections.Counter(), collections.Counter()
        points = sorted(boundaries)
        for a,b in zip(points, points[1:]):
            for kind, delta in boundaries[a]:
                active[kind] += delta
            if not active['gap']:
                continue
            primary = [k for k in ('compile','wait','submit') if active[k]]
            if len(primary) > 1:
                key = 'compile/wait/submit 并行交集（只计一次）'
            elif primary:
                key = {'compile':'仅 compileGraph', 'wait':'仅完成事件/流等待', 'submit':'仅提交 API'}[primary[0]]
            elif any(n for k,n in active.items() if k != 'gap'):
                key = '其余有 CPU/runtime span 的区间'
            else:
                key = '无细粒度 host span，已定位到此执行阶段'
            ledger[key] += (b-a)/count/1000
        expected = sum(b-a for a,b in spans)/count/1000
        assert abs(sum(ledger.values())-expected) < 1e-8
        results[phase] = {'gap_ms': expected, 'host_unions_ms': unions,
                          'exclusive_host_observation_ms': dict(ledger)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve(); out = root / 'target-gap-breakdown-01'
    summary = json.loads((out / 'host-inclusive-overlap.json').read_text())
    windows = json.loads((root / 'target-hardware-breakdown/selected-windows.json').read_text())
    details, rows = {}, []
    for rank in range(4):
        phases = json.loads((out / f'rank{rank}-phase-intervals.json').read_text())
        details[str(rank)] = analyze_phase_host(root, rank, phases)
    for pair in (0,1):
        total_gap = sum(summary[str(rank)]['gap_ms'] for rank in (pair,pair+2))
        gs = windows[str(pair)]
        target = sum((windows[str(pair+2)][g]['end_us']-w['start_us'])/1000 for g,w in gs.items())/len(gs)
        for rank in (pair,pair+2):
            for phase, value in summary[str(rank)]['phase_ms'].items():
                if not value:
                    continue
                rows.append(dict(tp_pair=pair, rank=rank, phase=phase, label=LABELS[phase],
                                 gap_ms=value, share_of_gap_pct=100*value/total_gap,
                                 share_of_target_pct=100*value/target))
        assert abs(sum(r['gap_ms'] for r in rows if r['tp_pair']==pair)-total_gap) < 1e-8
    with (out / 'phase-ledger.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out / 'phase-runtime-details.json').write_text(json.dumps(details,indent=2))
    report = ['# DSpark C6：122.892677 ms 未覆盖区间继续拆解', '',
              '复用同一份已保存的四卡 trace，没有重新压测。15 次 C6 丢弃前 2 次，generation 4–16 共 13 次；所有主表数值均摊到这 13 次。单位 ms/C6，不能除以六后当成实测 ms/token。', '',
              '先对每张卡同一 Target 完成标记窗口内的 TPC/MME/DMA 活动取并集，再求补集；将补集与实测 CPU 分组、原生重放入口和流同步边界求交。下面是**空档发生在哪个执行阶段**的互斥账本，不是假定该阶段独占 CPU，也不是可消除延迟预测。两 PP stage 之间原有 0.135776 ms 交接不属于此补集。', '']
    for pair in (0,1):
        selected = sorted([r for r in rows if r['tp_pair']==pair],key=lambda r:-r['gap_ms'])
        report += [f'## TP{pair} 互斥阶段账本', '', '| 执行阶段 | ms/C6 | 占本次未覆盖区间 | 占完整 Target |', '|---|---:|---:|---:|']
        for row in selected:
            report.append(f"| {row['label']} | {row['gap_ms']:.6f} | {row['share_of_gap_pct']:.3f}% | {row['share_of_target_pct']:.3f}% |")
        report += [f"| **合计** | **{sum(r['gap_ms'] for r in selected):.6f}** | **100%** | **{sum(r['share_of_target_pct'] for r in selected):.3f}%** |", '']
    report += ['## 大项内部：同区间观测继续去重', '',
               '“仅”指没有另两类 compile/wait/submit API 同时覆盖；可以与外层 Python/runtime span 并行。下表是阶段主表的子账，不得再次加入总账。后台线程上的等待不能直接认定为关键路径等待。', '']
    for rank, phase in [(0,'PP0 input preparation before native replay'),
                        (2,'PP1 snapshot and native capture before instantiate barrier'),
                        (2,'PP1 five group plan preparation and ordinary execution'),
                        (2,'PP1 queued work after final replay enqueue')]:
        value = details[str(rank)][phase]
        report += [f'### rank {rank}：{LABELS[phase]}', '', '| 去重后的观测 | ms/C6 |', '|---|---:|']
        for key, val in sorted(value['exclusive_host_observation_ms'].items(),key=lambda x:-x[1]):
            report.append(f'| {key} | {val:.6f} |')
        report += [f"| 合计 | {value['gap_ms']:.6f} |",'']
    report += ['## 已证实的优化入口', '',
       '1. **PP1 C6 重放输入合同失配。** PP wire 的 hidden stride 为 `(20496,5120,1)`，StageVariant 的 clone 为 `(20480,5120,1)`；FixedDecodeInputs.updates 按 stride 校验并返回 None，replay_native_decoder 随后全局失效 prepared plans。先修源布局与固定连续 staging 的绑定合同，保留 generation、地址、状态和依赖校验；不能通过删除 stride 检查放行错误的原生连续拷贝。',
       '2. **这次实际是 6 次五组重新准备、7 次五组缓存包装后再次捕获。** 五组包装区、快照/捕获区、实例化 stream barrier、同步后的计划组装区和状态恢复区均在本次 trace 中定位。相关 PP1 空档合计 82.852798 ms/C6（含失效/组间边界），占原 122.892677 ms 的 67.419%。这些是反复进入准备流程时的区间，不能宣布修复后会等额消失。',
       '3. **PP0 入口仍有 eager 小图。** 重放前 21.325249 ms 空档中，14.367912 ms 被 compileGraph 的跨线程并集覆盖；采集配置 PT_HPU_ENABLE_EAGER_CACHE=0。优先固定 embedding/residual/pre、Engram staging 和动态输入准备，纳入可重放的编译入口；不能把这个开关直接打开便声称数值和生命周期通过。',
       '4. **提交后的 18.605994 ms（PP0 6.739827 + PP1 11.866167）仍需按真实生产者依赖优化。** CPU 已可执行后续 verify/commit/draft 准备，因此部分 compile/copy span 与 Target 设备窗口重叠；这不是把 verify 或 PP commit 的 CPU span 全额重新算入 Target。PP1 其中 compileGraph 的交集并集为 8.272182 ms，也不能当成等量阻塞。', '',
       '## 归因边界', '',
       '- 最终阶段账本已全部对齐，但未全部定位到 C++ 子函数：捕获阶段有 21.582518 ms/C6 缺细粒度 host span，计划组装阶段有 8.847206 ms/C6 缺细粒度 host span。已知它们发生在重复捕获/实例化中，不能编造为 malloc、memcpy、HCL 或 Python 各占多少。下一版修复若仍有残余，仅需给 scheduleCapture / endCapture / HCL batch create / Synapse preparePlan / pipeline join 加范围计时，无需重复旧硬件 baseline。',
       '- 具体例子：rank2 generation6，在 instantiate stream synchronize 返回后存在 14.297313 ms 连续空档；原始 trace 只有外围 Target annotation，没有子调用、TPC/MME/DMA 事件。保存了原始事件检查，禁止凭父范围填造子调用耗时。',
       '- 缺 TPC/MME/DMA 事件不等于全芯片物理空闲；NIC/队列执行和未完整记录的事件仍可能存在。没有用等待时长推导 HBM 带宽，也没有把 compute utilization 当成 FLOP 效率。',
       '- 本采集带 profiler、阶段诊断和计划转储，不能用于 <50 ms 无 profiler 验收。该目标仍未达到。', '',
       '## 边界及来源', '',
       '- PP0：设备 stage_model_start → native_decoder_enqueue；之后至 stage_target_done。',
       '- PP1 重捕获：最后一个 Target Torch-Compiled Region 返回 → 实例化 synchronizeStream 开始 → synchronizeStream 返回 → 第一笔状态恢复 copy_ → native_decoder_enqueue 返回 → stage_target_done。',
       '- PP1 重准备：五个 Target Torch-Compiled Region 实测范围；区间外作为失效/输入/组间边界；没有把 compiled wrapper 时长当成 CPU 纯计算。',
       '- 源码：vllm_gaudi/ops/deepseek_v41_replay.py，tp2_graph_inputs.py，tp2_prepared_plan.py；tools/communication/tp2_native_decode_graph.h。本次运行 Python 源码来自 run/source 归档，运行时 fingerprint 来自 runtime-profile.json。',
       '- 输入：../target-hardware-breakdown/selected-windows.json、kernel-details.json.gz；../trace-analysis/rank*/host.jsonl.gz。逐代边界和补集在 rank*-phase-intervals.json。',
       '- host-inclusive-overlap.json 保留完整并行组合和每个 host API/线程的交集并集；phase-runtime-details.json 保留每个阶段的 API 类别子账。',
       '- PP0→PP1 完整 Target 保持 199.517769 ms/C6；只展开原 122.892677 ms 行，不改变先前实测设备活动账本。', '']
    (out / 'REPORT.md').write_text('\n'.join(report))
    sources = out / 'analysis-source'; sources.mkdir(exist_ok=True)
    for name in ('report_deepseek_v41_target_gaps.py','render_deepseek_v41_target_gaps.py','report_deepseek_v41_target_hardware.py'):
        shutil.copy2(Path(__file__).parent/name, sources/name)
    hashes = {}
    for path in [out/'REPORT.md',out/'phase-ledger.csv',out/'phase-runtime-details.json',*sources.iterdir()]:
        with path.open('rb') as f:
            hashes[str(path.relative_to(out))] = hashlib.file_digest(f,'sha256').hexdigest()
    (out/'sha256.json').write_text(json.dumps(hashes,indent=2))
    print(out/'REPORT.md')


if __name__ == '__main__':
    main()
