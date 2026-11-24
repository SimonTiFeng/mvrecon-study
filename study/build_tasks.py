#!/usr/bin/env python
"""Build the multi-view task set from the real-scan visibility graph.

Grouping rule (computed from GT scan visibility only, before any method runs):
  G1 high direct overlap : pair overlap >= HI
  G2 low  direct overlap : LO <= pair overlap < HI
  G3 bridging only       : pair overlap < LO and some candidate view M has
                           overlap(A,M) >= LO and overlap(M,B) >= LO
  G4 not connected       : pair overlap < LO and no bridge among the candidates

Task families:
  P2      two views {canonical, target}; kind G1/G2/G3/G4.
  MULTI   three or four views {A, B, ...} where the first two are a G3 pair and the extra
          view(s) include a real bridge; members are the bridge view(s), and the canonical
          view is one end of the pair.  For each G3 pair the bridge list is the *smallest*
          bridging view set (size 1 preferred).
  NEG     a G4 pair whose image content also has no overlap (checked with SIFT on the real
          images, threshold NEG_MATCHES) so "correct decision is to abstain" is
          well-founded.  The agent is not told the group.

Output: E:/mvrecon_real/tasks/tasks.json   and per task  E:/mvrecon_real/tasks/<task>/.
The canonical view is the reference; each task lists EVERY view's RGB and its local cloud
file, but no pose and no overlap label.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SC = pathlib.Path('E:/mvrecon_real/scenes')
OUT = pathlib.Path('E:/mvrecon_real/tasks')
HI = 0.20
LO = 0.03
NEG_MATCHES = 30
MAX_P2 = {'G1_high': 6, 'G2_low': 6, 'G3_bridge': 4, 'G4_unconnected': 4}
MAX_MULTI_PER_SCENE = 3
CANDIDATES = 10      # every scene has 10 study views


def sift_matches(scene, a, b):
    ma = json.loads((SC / scene / 'meta.json').read_text())
    rgb = {v['sid']: v['rgb'] for v in ma['views']}
    def g(s):
        im = cv2.imread(str(SC / scene / 'rgb' / rgb[s]), cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02).detectAndCompute(im, None)
    k1, d1 = g(a)
    k2, d2 = g(b)
    if d1 is None or d2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(d1, d2, k=2)
    good = [x for x, y in m if x.distance < 0.8 * y.distance]
    m2 = bf.knnMatch(d2, d1, k=2)
    back = {x.queryIdx: x.trainIdx for x, y in m2 if x.distance < 0.8 * y.distance}
    good = [x for x in good if back.get(x.trainIdx) == x.queryIdx]
    return len(good)


def main(scenes):
    g = json.loads((ROOT / 'study' / 'real_meta' / 'visibility_graph.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for scene in scenes:
        if scene not in g:
            continue
        pairs = g[scene]['pairs']
        views = g[scene]['views']
        by_group = {}
        for p in pairs:
            by_group.setdefault(p['group'], []).append(p)
        # P2 tasks
        for grp, cap in MAX_P2.items():
            sel = sorted(by_group.get(grp, []), key=lambda p: -p['overlap'])
            if grp == 'G4_unconnected':
                # need real image evidence of no overlap; scan overlap is already ~0
                keep = []
                for p in sel:
                    n = sift_matches(scene, p['a'], p['b'])
                    if n <= NEG_MATCHES:
                        keep.append({**p, 'sift_matches': n})
                    if len(keep) >= cap:
                        break
                sel = keep
            else:
                sel = sel[:cap]
            for p in sel:
                tasks.append({'task': '%s_%s_%s_%s' % (scene, p['group'][:2], p['a'][-4:], p['b'][-4:]),
                              'scene': scene, 'kind': 'P2', 'group': p['group'],
                              'canonical': p['a'], 'members': [p['a'], p['b']],
                              'gt_overlap': p['overlap'],
                              'gt_bridges': p.get('bridges', [])})
        # MULTI bridging tasks
        bridges = sorted(by_group.get('G3_bridge', []), key=lambda p: -p['overlap'])
        used = set()
        n_multi = 0
        for p in bridges:
            if n_multi >= MAX_MULTI_PER_SCENE:
                break
            br = p['bridges']
            if not br or p['a'] in used or p['b'] in used:
                continue
            members = [p['a'], p['b'], br[0]]
            key = tuple(sorted(members))
            used.add(p['a'])
            used.add(p['b'])
            n_multi += 1
            tasks.append({'task': '%s_G3M_%s_%s_%s' % (scene, p['a'][-4:], p['b'][-4:], br[0][-4:]),
                          'scene': scene, 'kind': 'MULTI', 'group': 'G3_bridge',
                          'canonical': p['a'], 'members': members,
                          'gt_overlap': p['overlap'], 'gt_bridges': br})
    # full candidate view list for every task (all 10 study views of the scene)
    for t in tasks:
        ma = json.loads((SC / t['scene'] / 'meta.json').read_text())
        t['candidates'] = [v['sid'] for v in ma['views']]
    (OUT / 'tasks.json').write_text(json.dumps(tasks, indent=1))
    from collections import Counter
    print('tasks', len(tasks), Counter((t['scene'], t['kind'], t['group']) for t in tasks))
    print(Counter(t['kind'] for t in tasks))
    return tasks


if __name__ == '__main__':
    main(sys.argv[1:] or ['office', 'playground', 'terrace', 'courtyard'])
