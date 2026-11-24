"""Search the lateral plane for the offset that maximises co-visibility agreement.

The fixed-point loop failed because it only uses points that survive at the CURRENT pose, so
it reconfirms itself. But selfconsistency_width.py showed the agreement fraction is strongly
peaked at the true offset for some targets (electro: 0.766 true vs 0.095 at the estimate),
i.e. the information IS in the agreement fraction -- it just has to be SEARCHED, not iterated.

This does a coarse-to-fine 2D scan over translation offsets in the reference frame (x,y),
rotation and z held at the baseline estimate, maximising
    agreement = fraction of reference points that project in-frame AND whose predicted depth
                matches the moving view's depth model within tau.

Two quantities matter, and the second is the honest one:
  - where the peak is (does the search find the true offset?)
  - how wide the near-peak set is (a broad plateau means the answer is not identifiable,
    and the correct behaviour is to REFUSE rather than return the plateau's centre)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
TAU = 0.10
SPAN = 6.0


def make_scorer(p_w, K, depth, R, t_base):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def score(dxy):
        t = t_base + np.array([dxy[0], dxy[1], 0.0])
        p_m = (p_w - t) @ R
        z = p_m[:, 2]
        ok = z > 0.2
        u = np.full(len(p_w), np.nan)
        v = np.full(len(p_w), np.nan)
        u[ok] = fx * p_m[ok, 0] / z[ok] + cx
        v[ok] = fy * p_m[ok, 1] / z[ok] + cy
        frame = ok & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
        cand = np.where(frame)[0]
        if len(cand) == 0:
            return 0.0
        iu = np.clip(np.round(u[cand]).astype(int), 1, W - 2)
        iv = np.clip(np.round(v[cand]).astype(int), 1, H - 2)
        zm = depth[iv, iu].astype(float)
        good = np.isfinite(zm) & (zm > 0)
        rel = np.abs(zm - z[cand]) / np.maximum(z[cand], 1e-6)
        return float(np.sum(good & (rel < TAU))) / len(p_w)

    return score


def main():
    rows = []
    rng = np.random.default_rng(0)
    for x in BASE:
        if x['method'] != 'A_icp':
            continue
        if x['score_corrected']['rot_err_deg'] >= 20:
            continue
        scene, tag, sid = x['scene'], x['tag'], x['sid']
        wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
        setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
        ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
        M = np.asarray(x['matrix_est'], float)
        R_est = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
        T = np.asarray(x['matrix_gt'], float)
        p_all = np.asarray(ref_c.as_o3d().points)
        p_w = p_all[rng.choice(len(p_all), min(2500, len(p_all)), replace=False)]

        t_base = M[:3, 3].copy()
        true_d = (T[:3, 3] - t_base)[:2]
        score = make_scorer(p_w, mov_c.K, mov_c.depth, R_est, t_base)

        # coarse-to-fine
        best = (0.0, 0.0, 0.0)
        step, span, centre = 0.75, SPAN, np.zeros(2)
        for _ in range(3):
            n = int(span / step)
            found = False
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    d = centre + np.array([i * step, j * step])
                    if np.linalg.norm(d) > SPAN + 1e-9:
                        continue
                    s = score(d)
                    if s > best[0]:
                        best, found = (s, d[0], d[1]), True
            centre = np.array([best[1], best[2]])
            step, span = step / 5, step
        # plateau: fraction of grid points within 95% of the peak
        near = []
        grid = np.arange(-SPAN, SPAN + 0.5, 0.5)
        for a in grid:
            for b in grid:
                if np.hypot(a, b) > SPAN:
                    continue
                s = score(np.array([a, b]))
                if s >= 0.95 * best[0] and best[0] > 0:
                    near.append((a, b))
        near = np.array(near) if near else np.zeros((0, 2))
        plateau_area = float(len(near)) * 0.25

        d_est = np.array([best[1], best[2]])
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'lateral_before': float(np.linalg.norm(true_d)),
                     'peak_agree': best[0],
                     'search_ctr_err': float(np.linalg.norm(d_est)),
                     'search_ans_err': float(np.linalg.norm(d_est - true_d)),
                     'true_in_plateau': bool(len(near) and
                                             np.min(np.linalg.norm(near - true_d, axis=1)) < 0.6),
                     'plateau_area_m2': plateau_area})
        r = rows[-1]
        print('%-14s%-4s%-12s lat %5.2f | peak %.3f  ans %5.2f m (was %5.2f)  plateau %5.1f m2  truth-in %s'
              % (scene, tag, sid, r['lateral_before'], r['peak_agree'], r['search_ans_err'],
                 r['lateral_before'], plateau_area, 'Y' if r['true_in_plateau'] else 'n'),
              flush=True)

    p = ROOT / 'out' / 'lateral_search.json'
    p.write_text(json.dumps(rows, indent=1))
    print('\nn=%d' % len(rows))
    before = np.array([r['lateral_before'] for r in rows])
    after = np.array([r['search_ans_err'] for r in rows])
    print('median lateral  %.2f -> %.2f m' % (np.median(before), np.median(after)))
    print('under 0.5 m     %d/%d -> %d/%d'
          % (int((before < 0.5).sum()), len(before), int((after < 0.5).sum()), len(after)))
    print('improved        %d/%d   worsened %d/%d'
          % (int((after < before).sum()), len(rows), int((after > before).sum()), len(rows)))
    print('truth inside reported plateau: %d/%d  (a refusal is correct on the rest)'
          % (int(sum(r['true_in_plateau'] for r in rows)), len(rows)))
    print('median plateau area %.1f m2' % np.median([r['plateau_area_m2'] for r in rows]))
    print('wrote', p)


if __name__ == '__main__':
    main()
