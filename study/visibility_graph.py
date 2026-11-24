#!/usr/bin/env python
"""Visibility graph, overlap and bridging statistics from REAL scan local clouds.

Each view V has a local cloud C_V of scan points visible to V (built by
build_real_local_clouds.py).  For a pair (A,B):
  overlap(A,B)   = fraction of A's cloud points that lie within EPS of B's cloud
                   (and symmetrically).  Uses scan points, not mono depth.
  bridge(A,B)    = exists a candidate view M with overlap(A,M) >= LO and
                   overlap(M,B) >= LO, while overlap(A,B) < LO.

Groups (rule fixed here, applied to all scenes):
  G1 high direct overlap : overlap >= HI
  G2 low direct overlap  : LO <= overlap < HI
  G3 bridging only       : overlap < LO and bridge exists
  G4 not connected       : overlap < LO and no bridge within the candidate set

Writes study/real_meta/visibility_graph.json and prints per-scene tables.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
SC = pathlib.Path('E:/mvrecon_real/scenes')
OUT = ROOT / 'study' / 'real_meta'
EPS = 0.06            # m, "same surface patch" tolerance
HI = 0.20             # high-overlap threshold
LO = 0.03             # low-overlap threshold


def load(scene, sid):
    d = np.load(SC / scene / (sid + '.npz'))
    return d['points'].astype(np.float64)


def main(scenes):
    OUT.mkdir(exist_ok=True)
    allg = {}
    for scene in scenes:
        meta = json.loads((SC / scene / 'meta.json').read_text())
        sids = [v['sid'] for v in meta['views']]
        P = {s: load(scene, s) for s in sids}
        T = {s: cKDTree(p) for s, p in P.items()}
        ov = {}
        for a in sids:
            for b in sids:
                if a == b:
                    continue
                d, _ = T[b].query(P[a], k=1)
                ov[(a, b)] = float((d < EPS).mean())
        sym = {(a, b): 0.5 * (ov[(a, b)] + ov[(b, a)])
               for a in sids for b in sids if a < b}
        pairs = []
        for (a, b), o in sym.items():
            bridges = [m for m in sids if m not in (a, b)
                       and min(ov[(a, m)], ov[(m, a)]) >= LO
                       and min(ov[(b, m)], ov[(m, b)]) >= LO]
            if o >= HI:
                g = 'G1_high'
            elif o >= LO:
                g = 'G2_low'
            elif bridges:
                g = 'G3_bridge'
            else:
                g = 'G4_unconnected'
            pairs.append({'a': a, 'b': b, 'overlap': o, 'group': g,
                          'bridges': bridges,
                          'explicit_link': o >= LO})
        allg[scene] = {'views': sids, 'pairs': pairs,
                       'thresholds': {'eps_m': EPS, 'hi': HI, 'lo': LO}}
        from collections import Counter
        c = Counter(p['group'] for p in pairs)
        print('\n== %s  %d views, %d pairs: %s' % (scene, len(sids), len(pairs), dict(c)))
        top = sorted(pairs, key=lambda p: -p['overlap'])[:8]
        for p in top:
            print('  %s %s %.3f %s' % (p['a'], p['b'], p['overlap'], p['group']))
        for p in pairs:
            if p['group'] == 'G3_bridge':
                print('  BRIDGE %s -> %s via %s (ov %.3f)' % (p['a'], p['b'], p['bridges'][:3], p['overlap']))
                break
    (OUT / 'visibility_graph.json').write_text(json.dumps(allg, indent=1))
    print('\nwrote', OUT / 'visibility_graph.json')


if __name__ == '__main__':
    main(sys.argv[1:] or ['office', 'courtyard', 'playground', 'terrace'])
