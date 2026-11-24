#!/usr/bin/env python
"""Paired tests for agent vs each traditional baseline on the 40 instances."""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'out' / 'agent_study_20261123'


def boot_ci(x, n=20000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return [None, None]
    rng = np.random.default_rng(seed)
    med = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def main():
    base = json.loads((OUT / 'eval' / 'baselines.json').read_text())
    agent = json.loads((OUT / 'eval' / 'agent_results.json').read_text())
    b = {('%s_%s_%s' % (r['scene'], r['tag'], r['sid']), r['method']): r for r in base}
    a = {r['case']: r for r in agent}
    out = {'n_pairs': 0, 'comparisons': {}}
    for m in sorted({r['method'] for r in base}):
        dr, dl, rows = [], [], []
        b_win = a_win = 0
        for c, ar in a.items():
            br = b.get((c, m))
            if not br or 'rot_err' not in br or 'rot_err' not in ar:
                continue
            rows.append((ar['rot_err'], ar['lat_err'], br['rot_err'], br['lat_err'],
                         ar['success'], br['success']))
        if not rows:
            continue
        arr = np.array(rows, float)
        dr = arr[:, 0] - arr[:, 2]
        dl = arr[:, 1] - arr[:, 3]
        try:
            wr = stats.wilcoxon(arr[:, 0], arr[:, 2], zero_method='wilcox')
            wl = stats.wilcoxon(arr[:, 1], arr[:, 3], zero_method='wilcox')
            p_rot, p_lat = float(wr.pvalue), float(wl.pvalue)
        except Exception:
            p_rot = p_lat = None
        # McNemar on success
        b01 = int(((arr[:, 4] == 1) & (arr[:, 5] == 0)).sum())
        b10 = int(((arr[:, 4] == 0) & (arr[:, 5] == 1)).sum())
        if b01 + b10:
            p_succ = float(stats.binomtest(b01, b01 + b10, 0.5).pvalue)
        else:
            p_succ = None
        out['n_pairs'] = int(len(rows))
        out['comparisons'][m] = {
            'n': int(len(rows)),
            'rot_delta_median': float(np.median(dr)),
            'rot_delta_ci95': boot_ci(dr),
            'rot_wilcoxon_p': p_rot,
            'rot_agent_better': int((dr < -0.05).sum()),
            'rot_agent_worse': int((dr > 0.05).sum()),
            'lat_delta_median': float(np.median(dl)),
            'lat_delta_ci95': boot_ci(dl),
            'lat_wilcoxon_p': p_lat,
            'lat_agent_better': int((dl < -0.02).sum()),
            'lat_agent_worse': int((dl > 0.02).sum()),
            'success_agent_only': b01, 'success_baseline_only': b10,
            'mcnemar_exact_p': p_succ,
            'agent_successes': int(arr[:, 4].sum()),
            'baseline_successes': int(arr[:, 5].sum()),
        }
    (OUT / 'eval' / 'stats_tests.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
