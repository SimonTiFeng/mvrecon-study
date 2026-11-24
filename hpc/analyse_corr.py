"""Analyse the HPC2 correspondence sweep: accuracy, cost, cross-scene transfer, Pareto front."""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def med(v):
    return float(np.median(v)) if len(v) else float('nan')


def main():
    rows = load(sys.argv[1])
    out_md = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    lines = []
    def emit(s=''):
        print(s)
        lines.append(s)

    emit('# Correspondence sweep (HPC2)\n')
    emit('records: %d\n' % len(rows))

    # ---- 1. accuracy and cost at zero outliers
    base = [r for r in rows if r.get('outlier_frac', 0) == 0]
    g = collections.defaultdict(list)
    for r in base:
        g[(r['matcher'], r['mask'], r['solve'])].append(r)
    emit('## Accuracy and cost (no injected outliers)\n')
    emit('| matcher | mask | solve | n | rot med | rot<2° | lat med | lat<0.5m | cost med (ms) |')
    emit('|---|---|---|---|---|---|---|---|---|')
    pareto = []
    for k in sorted(g):
        v = g[k]
        rot = np.array([x['rot_err'] for x in v])
        lat = np.array([x['lat_err'] for x in v])
        cost = np.array([x['ms_detect'] + x['ms_match'] + x['ms_solve'] for x in v])
        emit('| %s | %s | %s | %d | %.2f° | %d | %.2f m | %d | %.0f |'
             % (k[0], k[1], k[2], len(v), med(rot), int((rot < 2).sum()),
                med(lat), int((lat < 0.5).sum()), med(cost)))
        pareto.append({'cfg': k, 'n': len(v), 'rot_med': med(rot),
                       'rot_lt2': int((rot < 2).sum()),
                       'lat_med': med(lat), 'lat_lt05': int((lat < 0.5).sum()),
                       'cost_med': med(cost)})

    # ---- 2. Pareto front: lower cost is better, lower lat_med is better
    front = []
    for a in pareto:
        dominated = any((b['cost_med'] <= a['cost_med'] and b['lat_med'] <= a['lat_med'])
                        and (b['cost_med'] < a['cost_med'] or b['lat_med'] < a['lat_med'])
                        for b in pareto)
        if not dominated:
            front.append(a)
    emit('\n## Pareto front (cost vs median lateral error)\n')
    emit('| matcher | mask | solve | lat med | cost med (ms) |')
    emit('|---|---|---|---|---|')
    for a in sorted(front, key=lambda z: z['cost_med']):
        emit('| %s | %s | %s | %.2f m | %.0f |'
             % (a['cfg'][0], a['cfg'][1], a['cfg'][2], a['lat_med'], a['cost_med']))

    # ---- 3. cross-scene transfer for the best configuration by median lateral error
    best = min(pareto, key=lambda z: (z['lat_med'], z['rot_med']))
    bk = best['cfg']
    emit('\n## Cross-scene transfer — %s / %s / %s\n' % bk)
    emit('| scene | n | rot med | lat med | lat<0.5m | baseline lat med |')
    emit('|---|---|---|---|---|---|')
    per = collections.defaultdict(list)
    for r in base:
        if (r['matcher'], r['mask'], r['solve']) == bk:
            per[r['scene']].append(r)
    for sc in sorted(per):
        v = per[sc]
        lat = np.array([x['lat_err'] for x in v])
        rot = np.array([x['rot_err'] for x in v])
        bl = np.array([x['base_trans'] for x in v])
        emit('| %s | %d | %.2f° | %.2f m | %d | %.2f m |'
             % (sc, len(v), med(rot), med(lat), int((lat < 0.5).sum()), med(bl)))

    # ---- 4. outlier robustness
    emit('\n## Robustness to gross outlier correspondences\n')
    emit('| solve | outlier | rot med | lat med | lat<0.5m |')
    emit('|---|---|---|---|---|')
    g2 = collections.defaultdict(list)
    for r in rows:
        g2[(r['solve'], r.get('outlier_frac', 0))].append(r)
    for k in sorted(g2):
        v = g2[k]
        rot = np.array([x['rot_err'] for x in v])
        lat = np.array([x['lat_err'] for x in v])
        emit('| %s | %.0f%% | %.2f° | %.2f m | %d |'
             % (k[0], 100 * k[1], med(rot), med(lat), int((lat < 0.5).sum())))

    if out_md:
        out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print('\nwrote', out_md)


if __name__ == '__main__':
    main()
