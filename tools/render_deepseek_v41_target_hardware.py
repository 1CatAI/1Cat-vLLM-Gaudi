# SPDX-License-Identifier: Apache-2.0
"""Publish functional Target bills and resource duty from retained intervals."""
import argparse
import collections
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
from report_deepseek_v41_target_hardware import classify, duration, partition


def signature(c):
    return json.dumps([[dict(shape=t['shape'],dtype=t['dtype'],location=t['location']) for t in c[side]]
                       for side in ('inputs','outputs')],sort_keys=True)


def calls(row):
    """TPC compiled-node calls from exact recipe ROI work-engine counts.

    A node may contain multiple TPC ROIs. One full node execution must contain
    every ROI's working-engine packet, and consecutive executions must not
    overlap. MME timing uses its lane-complete equal-count cohorts only when
    all observed participants agree and consecutive cohorts do not overlap.
    No union/count quotient is used as a call duration.
    """
    c=row.get('contract');out=[];count=0
    if not c:return None,None,'no compiled compute contract',[]
    for obs in row['observations']:
        es=sorted(obs['events'],key=lambda e:e[4]);cohorts=[]
        if row['engine']=='TPC':
            expected=sum(c['symbol']['working_engines'])
            if not expected or len(es)%expected:return None,None,'ROI packet count incomplete',[]
            cohorts=[es[i:i+expected] for i in range(0,len(es),expected)]
        elif row['engine']=='MME':
            lanes=collections.defaultdict(list)
            for e in es:lanes[(e[2],e[3])].append(e)
            if len({len(v) for v in lanes.values()})!=1:return None,None,'MME lane counts differ',[]
            cohorts=[[v[i] for v in lanes.values()] for i in range(len(next(iter(lanes.values()))))]
        else:return None,None,'DMA packets are reported separately',[]
        spans=[(min(e[4] for e in group),max(e[4]+e[5] for e in group)) for group in cohorts]
        if any(a2<b1 for (a1,b1),(a2,b2) in zip(spans,spans[1:])):
            return None,None,'concurrent cohorts cannot be uniquely reconstructed',[]
        if any(abs(e[0]-e[4])>0.002 or abs(e[1]-(e[4]+e[5]))>0.002 for e in es):
            return None,None,'call clipped by target boundary',[]
        out.extend((b-a)/1000 for a,b in spans);count+=len(spans)
    note=('compiled TPC node, all recipe ROIs and engine packets' if row['engine']=='TPC'
          else 'MME monitor cohort, equal ordered lane counts, nonoverlapping cohorts')
    return count,statistics.mean(out) if out else None,note,out


def canonical(group):
    return '跨模块融合与未归因计算' if group in ('跨模块融合','未完整归因的计算') else group


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path);args=p.parse_args();run=args.run.resolve()
    raw=run/'target-hardware-breakdown';out=args.output or run/'target-kernel-report';out.mkdir(exist_ok=False)
    summary=json.loads((raw/'summary.json').read_text());windows=json.loads((raw/'selected-windows.json').read_text())
    with gzip.open(raw/'kernel-details.json.gz','rt') as f:data=json.load(f)
    n=len(summary['generations']);contracts={};match=collections.Counter()
    for rank in range(4):
        items=json.loads((run/f'target-recipe-metadata/rank{rank}/enriched-contracts.json').read_text())
        contracts[rank]={(str(c['recipe_id']),c['symbol']['node']):c for c in items}
    spans={rank:{g:collections.defaultdict(list) for g in summary['generations']} for rank in range(4)}
    aggregated={};lite=[];decoded_sram=collections.Counter()
    pergen_calls={rank:{g:collections.Counter() for g in summary['generations']} for rank in range(4)}
    for index,row in enumerate(data):
        rank=row['rank'];c=contracts[rank].get((row['recipe'].split(':')[0],row['node']))
        if c:row['contract']=c;match[(rank,row['engine'],'matched')]+=1
        else:match[(rank,row['engine'],'unmatched')]+=1
        group,purpose=classify(row,c);group=canonical(group)
        row['group'],row['purpose']=group,purpose
        count,mean,note,latencies=calls(row)
        for obs in row['observations']:
            gen=obs['generation'];es=obs['events']
            spans[rank][gen][group].extend((e[0],e[1]) for e in es)
            if row['engine']=='TPC' and c:
                expected=sum(c['symbol']['working_engines'])
                if expected and len(es)%expected==0:
                    label=('sparse_attention' if 'sparse_attn' in row['kernel'] else
                           'expert_dequant' if 'mxfp4_prepared_dequant' in row['kernel'] else None)
                    if label:pergen_calls[rank][gen][label]+=len(es)//expected
        if c and 'mxfp4_prepared_dequant' in row['kernel']:
            for t in c['outputs']:decoded_sram[(rank,t['dtype'],t['location'])]+=1
        stable=re.sub(r'^(fused_kernel_0x[0-9A-F]+)_[0-9A-F]+_',r'\1_*_',row['kernel'])
        if row['engine']=='DMA':
            # API/command-page sequence values are invocation identifiers,
            # not different kernel algorithms. Retain every original name.
            stable=re.sub(r'apiId=\d+', 'apiId=*', stable)
            stable=re.sub(r'chunk=\d+', 'chunk=*', stable)
        sig=signature(c) if c else 'unknown'
        key=(rank,group,purpose,row['engine'],stable,sig)
        if key not in aggregated:
            aggregated[key]=dict(rank=rank,group=group,purpose=purpose,engine=row['engine'],kernel=stable,
                                 kernels=set(),contract=c,spans=collections.defaultdict(list),
                                 lane_packets=0,call_count=0,call_latencies=[],unknown_calls=False,node_refs=[])
        a=aggregated[key];a['kernels'].add(row['kernel']);a['lane_packets']+=row['lane_packets']
        if count is None:a['unknown_calls']=True
        else:a['call_count']+=count;a['call_latencies'].extend(latencies)
        for obs in row['observations']:a['spans'][obs['generation']].extend((e[0],e[1]) for e in obs['events'])
        a['node_refs'].append(index)
        record={k:v for k,v in row.items() if k!='observations'}
        record.update(raw_detail_index=index,physical_calls=count,mean_call_ms=mean,physical_call_note=note)
        lite.append(record)
    for rank,r in enumerate(summary['ranks']):
        groups=collections.Counter();exclusive=collections.Counter();sample=[]
        for gen,named in spans[rank].items():
            w=windows[str(rank)][str(gen)];a,b=w['start_us'],w['end_us']
            for group,s in named.items():groups[group]+=duration(s)
            for key,v in partition(named,a,b).items():
                exclusive[key[0] if len(key)==1 else '跨功能组重叠' if key else 'TPC/MME/DMA 未覆盖区间']+=v
            sample.append(dict(generation=gen,stage_window_ms=(b-a)/1000,**pergen_calls[rank][gen]))
        total=r['stage_window_ms']*n*1000
        def rows(counter):return [dict(name=k,ms_per_C6=v/n/1000,share_pct=v/total*100) for k,v in counter.most_common()]
        r['groups']=rows(groups);r['exclusive']=rows(exclusive);r['per_generation']=sample
        r['matched_compute_nodes']=sum(v for (rk,e,status),v in match.items() if rk==rank and e!='DMA' and status=='matched')
        r['unmatched_compute_nodes']=sum(v for (rk,e,status),v in match.items() if rk==rank and e!='DMA' and status=='unmatched')
        # Monitor lanes, not FLOP/bandwidth efficiency. The compute TPC mask
        # is verified by 24 executing lanes in this acquisition; the extra
        # reserved D0 TPC metadata lane never executes a model kernel.
        r['monitor_duty']={}
        for engine in ('TPC','MME','DMA'):
            ls=[x for x in r['lanes'] if x['engine']==engine]
            r['monitor_duty'][engine]=dict(observed_lanes=len(ls),mean_observed_lane_duty_pct=
                sum(x['active_ms_per_C6'] for x in ls)/len(ls)/r['stage_window_ms']*100 if ls else None)
    kernel_rows=[]
    for key,a in aggregated.items():
        active=sum(duration(s) for s in a.pop('spans').values())/n/1000
        c=a.pop('contract');lat=a.pop('call_latencies');unknown=a.pop('unknown_calls');count=a.pop('call_count')
        a['kernels']=sorted(a['kernels'])
        a.update(activity_ms_per_C6=active,stage_share_pct=active/summary['ranks'][a['rank']]['stage_window_ms']*100,
                 calls_per_C6=None if unknown else count/n,observed_calls=None if unknown else count,
                 mean_call_ms=None if unknown else statistics.mean(lat) if lat else None,
                 io_contract=json.loads(key[-1]) if c else None,
                 graph=c['graph']['path'] if c else None)
        kernel_rows.append(a)
    kernel_rows.sort(key=lambda a:(a['rank'],-a['activity_ms_per_C6']))
    # Host events annotate CPU work; they never replace measured hardware.
    for rank,r in enumerate(summary['ranks']):
        host_summary=[]
        events=[]
        with gzip.open(run/f'trace-analysis/rank{rank}/host.jsonl.gz','rt') as f:
            for line in f:
                ts,d,pid,tid,cat,name=json.loads(line)
                if ('Torch-Compiled Region' in name or name=='compileGraph' or
                    name.startswith('enqueueWithExternalEvents') or name.startswith('v41::')):
                    events.append((ts,ts+d,cat,name))
        for gen in summary['generations']:
            w=windows[str(rank)][str(gen)];h=w['record']['host'];offset=w['host_to_trace_us']
            a=h['round_start']/1000+offset;b=h['target_submit_done']/1000+offset
            compiled=[e for e in events if 'Torch-Compiled Region' in e[3] and a<=e[0]<b]
            compiles=[e for e in events if e[3]=='compileGraph' and a<=e[0]<b]
            enqueue=[e for e in events if e[3].startswith('enqueueWithExternalEvents') and a<=e[0]<b]
            host_summary.append(dict(generation=gen,compiled_group_calls=len(compiled),
                compiled_group_cpu_union_ms=duration((e[0],e[1]) for e in compiled)/1000,
                compileGraph_calls=len(compiles),compileGraph_cpu_union_ms=duration((e[0],e[1]) for e in compiles)/1000,
                enqueue_events=len(enqueue)))
        r['host_per_generation']=host_summary
    summary['decoded_outputs']=[dict(rank=k[0],dtype=k[1],location=k[2],nodes=v) for k,v in decoded_sram.items()]
    summary['kernel_role_shape_rows']=len(kernel_rows)
    summary['count_contract']='TPC: serialized recipe ROI engine totals, ordered full-node cohorts. MME: observed equal-count ordered monitor cohorts. Incomplete/overlapping cohorts retain null; packet counts are separate.'
    summary['source_scripts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('report_deepseek_v41_target_hardware.py'),Path(__file__).with_name('enrich_deepseek_v41_target_contracts.py'))}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'kernel-summary.json').write_text(json.dumps(kernel_rows,ensure_ascii=False,indent=2)+'\n')
    with gzip.open(out/'node-details.json.gz','wt') as f:json.dump(lite,f,ensure_ascii=False)
    with (out/'kernel-summary.csv').open('w') as f:
        fields=['rank','group','purpose','engine','kernel','activity_ms_per_C6','stage_share_pct','calls_per_C6','observed_calls','mean_call_ms','lane_packets','io_contract','graph']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(kernel_rows)
    md=['# DSpark C6 Target：功能与显卡资源拆解','',
        f'本次四卡硬件 trace；64-token 请求，15 次完整 C6，丢弃前 2 次，保留 {n} 次，generation {summary["generations"][0]}–{summary["generations"][-1]}。单位 ms/C6（一次六位置 Target，不是 ms/token）。',
        '同次采集以设备完成标记划界，覆盖两 PP stage 的 40 层；Target 不含词表 head、accepted-prefix、draft 和 PP commit。Profiler 和重复捕获都影响本窗口，不能当成无 profiler 基线或用于跨采集相减。','',
        '功能表给出互斥时间，跨功能重叠单列；未被 TPC/MME/DMA 事件覆盖的区间不自动认定为 CPU、通信或可消除空闲。NIC 带宽与 HBM 带宽未采集。','']
    for pair in summary['pairs']:
        tp=pair['tp'];total=pair['mean']['total_ms'];ex=collections.Counter();un=collections.Counter()
        for rank in (tp,tp+2):
            ex.update({x['name']:x['ms_per_C6'] for x in summary['ranks'][rank]['exclusive']})
            un.update({x['name']:x['ms_per_C6'] for x in summary['ranks'][rank]['groups']})
        ex['PP0 完成→PP1 开始']=pair['mean']['handoff_ms']
        assert abs(sum(ex.values())-total)<0.00001
        pair['functional_bill']=[dict(group=k,exclusive_ms=v,share_pct=v/total*100,activity_union_ms=un.get(k)) for k,v in ex.most_common()]
        md += [f'## TP{tp} 完整 Target：{total:.6f} ms/C6','', '| 功能组 | 互斥 ms/C6 | 占完整 Target | 活动并集 ms/C6（可重叠） |','|---|---:|---:|---:|']
        for x in pair['functional_bill']:
            uv=f"{x['activity_union_ms']:.6f}" if x['activity_union_ms'] is not None else '—'
            md.append(f"| {x['group']} | {x['exclusive_ms']:.6f} | {x['share_pct']:.3f}% | {uv} |")
        md += [f'| **合计** | **{total:.6f}** | **100%** | — |','']
    md += ['## 四卡资源占用','',
           '活动率表示窗口内该引擎至少一个监测 lane 活动的时间并集；引擎间可重叠。lane 平均值仅为监测点占空比，不是峰值 FLOPS 或 HBM 带宽效率。','',
           '| rank | stage ms/C6 | TPC 活动 ms / % | MME 活动 ms / % | DMA 活动 ms / % | 三引擎未覆盖 ms |','|---:|---:|---:|---:|---:|---:|']
    for r in summary['ranks']:
        e={x['name']:x for x in r['engines']};gap=next(x['ms_per_C6'] for x in r['exclusive'] if x['name']=='TPC/MME/DMA 未覆盖区间')
        fmt=lambda k:f"{e[k]['ms_per_C6']:.6f} / {e[k]['share_pct']:.3f}%"
        md.append(f"| {r['rank']} | {r['stage_window_ms']:.6f} | {fmt('TPC')} | {fmt('MME')} | {fmt('DMA')} | {gap:.6f} |")
    md += ['', '## 调用与数据流审计','',
           'TPC 调用依据本次 recipe 的 working_engines/ROI 包数重建；MME 按同节点各 lane/WB 监测点的等数顺序组重建。跨组重叠或不完整包保留未知。调用均值来自各完整组首尾实测，未用活动并集除以调用数。',
           '每节点完整形状、精度、SRAM/DRAM、源图、融合来源和原始事件索引保存在 node-details.json.gz；全部 kernel-role-shape 行在 kernel-summary.csv/JSON。','']
    for r in summary['ranks']:
        md += [f"rank {r['rank']}: compute nodes matched {r['matched_compute_nodes']}, unresolved {r['unmatched_compute_nodes']}; TPC lanes {r['monitor_duty']['TPC']['observed_lanes']}, MME lanes {r['monitor_duty']['MME']['observed_lanes']}."]
    md += ['', '## 逐 kernel 功能明细','',
           '| rank | 功能组 / 具体工作 | Kernel | 输入→输出精度/形状 | 活动 ms/C6 | stage 占比 | 次/C6 | 完整调用均值 ms | lane 包 |','|---:|---|---|---|---:|---:|---:|---:|---:|']
    for a in kernel_rows:
        io=a['io_contract'];contract=(' ; '.join(str(t['shape'])+' '+t['dtype']+' '+t['location'] for t in io[0])+' → '+' ; '.join(str(t['shape'])+' '+t['dtype']+' '+t['location'] for t in io[1])) if io else '未记录张量合同（DMA 命令/拷贝）'
        ct=f"{a['calls_per_C6']:.3f}" if a['calls_per_C6'] is not None else '未知'
        mean=f"{a['mean_call_ms']:.6f}" if a['mean_call_ms'] is not None else '未知'
        md.append(f"| {a['rank']} | {a['group']} / {a['purpose']} | `{a['kernel']}` | {contract} | {a['activity_ms_per_C6']:.6f} | {a['stage_share_pct']:.3f}% | {ct} | {mean} | {a['lane_packets']} |")
    (out/'REPORT.md').write_text('\n'.join(md)+'\n')
    # Include final bills after their reconciliation.
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DSpark Target kernel report</title><style>body{font:14px system-ui;margin:24px;background:#fafafa;color:#222}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{position:sticky;top:0;background:#eee}input,select{padding:8px;margin:6px}pre{white-space:pre-wrap;max-width:1000px}summary{cursor:pointer}</style><h1>DSpark C6 Target 硬件拆解</h1><p>13 次 C6，所有时间 ms/C6。活动并集可重叠；完整互斥分区见 <a href="REPORT.md">报告</a>。输入、输出、SRAM/DRAM 与源图可展开。</p><input id="q" placeholder="搜索 kernel / 功能"><select id="rank"><option value="">全部 rank</option><option>0</option><option>1</option><option>2</option><option>3</option></select><table><thead><tr><th>rank/功能</th><th>kernel 与来源</th><th>活动 ms/C6</th><th>占 stage</th><th>次/C6</th><th>平均调用 ms</th></tr></thead><tbody id="rows"></tbody></table><script>const data=DATA;const esc=s=>String(s??'未知').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function draw(){const q=document.querySelector('#q').value.toLowerCase(),r=document.querySelector('#rank').value;document.querySelector('#rows').innerHTML=data.filter(x=>(!r||String(x.rank)===r)&&JSON.stringify(x).toLowerCase().includes(q)).map(x=>`<tr><td>${x.rank} / ${esc(x.group)}<br>${esc(x.purpose)}</td><td><details><summary>${esc(x.kernel)}</summary><pre>${esc(JSON.stringify({kernels:x.kernels,io:x.io_contract,graph:x.graph,nodes:x.node_refs},null,2))}</pre></details></td><td>${x.activity_ms_per_C6.toFixed(6)}</td><td>${x.stage_share_pct.toFixed(3)}%</td><td>${x.calls_per_C6===null?'未知':x.calls_per_C6.toFixed(3)}</td><td>${x.mean_call_ms===null?'未知':x.mean_call_ms.toFixed(6)}</td></tr>`).join('')}document.querySelector('#q').oninput=draw;document.querySelector('#rank').onchange=draw;draw();</script></html>'''
    (out/'TRACE.html').write_text(page.replace('DATA',json.dumps(kernel_rows,ensure_ascii=False).replace('</','<\\/')))
    print(json.dumps(dict(output=str(out),rows=len(kernel_rows),pairs=[p['mean'] for p in summary['pairs']]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
