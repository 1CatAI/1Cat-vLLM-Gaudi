# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Per-token anatomy of a DSV4.1 decode raw trace (no kernel names needed).
Usage: trace_anatomy.py <trace-analysis-window/rankN dir>
Needs hardware.jsonl.gz, inventory.json, device-windows.json in that dir.
Reports: compute union (TPC-only / MME-only / overlap), idle split into
communication-adjacent waits vs compute->compute dependency gaps, launch counts,
and per-communication-segment table (attention side / MoE side / special)."""
import gzip, json, collections, bisect, statistics, sys, os
try:
    from .interval_index import IntervalIndex
except ImportError:
    from interval_index import IntervalIndex
D = sys.argv[1]
nodes = json.load(open(os.path.join(D, 'inventory.json')))['nodes']
win = json.load(open(os.path.join(D, 'device-windows.json')))['windows_us']
starts = [w[0] for w in win]
ev = collections.defaultdict(list); lanecnt = collections.defaultdict(collections.Counter)
for l in gzip.open(os.path.join(D, 'hardware.jsonl.gz'), 'rt'):
    ts, dur, lane, node, kind, api = json.loads(l)
    eng = lane.split('] ')[-1]
    g = 'HCL' if 'HCL' in eng else 'TPC' if eng.startswith('TPC') else 'MME' if eng.startswith('MME') else 'DMA'
    null = node < len(nodes) and nodes[node]['kernel'] == 'null'
    if null: g += '_null'
    ev[g].append((ts, ts + dur))
    i = bisect.bisect_right(starts, ts) - 1
    if i >= 0 and ts < win[i][1] and not null and g in ('TPC', 'MME'): lanecnt[(i, g)][lane] += 1
for g in ev:
    ev[g] = IntervalIndex(ev[g])
def union(iv, lo, hi):
    if isinstance(iv, IntervalIndex):
        return iv.clip(lo, hi)
    out = []
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in iv if b > lo and a < hi):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
tot = lambda u: sum(b - a for a, b in u)
def inter(u, v):
    i = j = 0; s = 0
    while i < len(u) and j < len(v):
        a = max(u[i][0], v[j][0]); b = min(u[i][1], v[j][1])
        if b > a: s += b - a
        if u[i][1] < v[j][1]: i += 1
        else: j += 1
    return s
def clusters(u, tol):
    out = []
    for a, b in u:
        if out and a - out[-1][1] <= tol: out[-1][1] = b
        else: out.append([a, b])
    return out
n = len(win); R = collections.defaultdict(float)
gap = collections.defaultdict(lambda: [0, 0.0])
seg = collections.defaultdict(lambda: collections.defaultdict(float)); segc = collections.Counter()
for lo, hi in win:
    U = {g: union(ev[g], lo, hi) for g in ('TPC', 'MME', 'HCL', 'DMA')}
    comp = union(U['TPC'] + U['MME'], lo, hi)
    allact = union(comp + U['HCL'] + U['DMA'], lo, hi)
    R['period'] += hi - lo; R['compute'] += tot(comp); R['TPC'] += tot(U['TPC']); R['MME'] += tot(U['MME'])
    R['HCL_DMA_only'] += tot(union(U['HCL'] + U['DMA'], lo, hi)) - inter(union(U['HCL'] + U['DMA'], lo, hi), comp)
    R['idle'] += (hi - lo) - tot(allact)
    blocks = []
    for a, b, g in sorted((a, b, g) for g in ('TPC', 'MME', 'HCL', 'DMA') for a, b in U[g]):
        if blocks and a <= blocks[-1][1]:
            if b >= blocks[-1][1]: blocks[-1][4] = g
            blocks[-1][1] = max(blocks[-1][1], b); blocks[-1][3].add(g)
        else: blocks.append([a, b, g, {g}, g])
    for x, y in zip(blocks, blocks[1:]):
        d = y[0] - x[1]
        if d <= 0: continue
        k = 'comm' if ('HCL' in x[3] or 'HCL' in y[3]) else f'{x[4]}->{y[2]}'
        gap[k][0] += 1; gap[k][1] += d
    H = clusters(union(ev['HCL'], lo, hi), 3.0); cuts = [lo] + [h[0] for h in H] + [hi]
    for a, b in zip(cuts, cuts[1:]):
        T = union(ev['TPC'], a, b); M = union(ev['MME'], a, b); C = union(T + M, a, b)
        if not C: continue
        c = tot(C); nt, nm = len(T), len(M)
        typ = 'attention-side' if (nm == 6 and 40 < c < 60) else 'moe-side' if (nm == 7 and 60 < c < 80) else 'special'
        s = seg[typ]; segc[typ] += 1
        s['period'] += b - a; s['compute'] += c; s['TPC'] += tot(T); s['MME'] += tot(M); s['idle'] += (b - a) - c
pt = lambda x: x / n / 1000
print(f"tokens {n}  period {pt(R['period']):.3f} ms  compute {pt(R['compute']):.3f} (TPC {pt(R['TPC']):.3f}, MME {pt(R['MME']):.3f}, overlap {pt(R['TPC']+R['MME']-R['compute']):.3f})")
print(f"no-engine idle {pt(R['idle']):.3f} ms  HCL/DMA-only {pt(R['HCL_DMA_only']):.3f} ms")
print("gaps between busy blocks (count/token, ms/token, mean us):")
for k, (c, s) in sorted(gap.items(), key=lambda x: -x[1][1]):
    print(f"  {k:12s} {c/n:7.1f} {s/n/1000:6.3f} {s/max(c,1):6.2f}")
for g in ('TPC', 'MME'):
    mx = statistics.mean(max(c.values()) for (i, e), c in lanecnt.items() if e == g)
    print(f"{g} launches/token (busiest lane, lower bound): {mx:.0f}")
print("segments between communication points:")
for t, s in seg.items():
    c = segc[t]
    print(f"  {t:15s} {c/n:5.1f}/token  per-seg period {s['period']/c:6.1f}us compute {s['compute']/c:5.1f} TPC {s['TPC']/c:5.1f} MME {s['MME']/c:5.1f} idle {s['idle']/c:5.1f}  | per-token {s['period']/n/1000:.3f} ms, idle {s['idle']/n/1000:.3f} ms")
