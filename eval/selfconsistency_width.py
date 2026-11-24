"""Why the GT-free co-visibility filter cannot localise: measure its discriminating width.

The estimated-pose loop converged to the WRONG pose and stayed there for 4 rounds
(courtyard 3.31 -> 3.28 m, electro 2.33 -> 2.35 m). Hypothesis: the co-visibility test
(depth agreement under the current pose) is satisfied over a whole RANGE of lateral offsets,
because a large planar surface looks locally identical when shifted.

Test: sweep the moving cloud's translation by delta along a lateral direction, and for each
delta measure the fraction of reference points that project in-frame AND whose predicted
depth agrees with the moving view's depth model within tau.

If that profile is flat/broad, the filter has no localising power and no amount of iterating
it will converge -- iteration preserves whatever the profile already admits. The correct
delta must be found by something OUTSIDE local geometric agreement.

Reported per target: the agreement profile, its width at 90% of maximum, and where the true
offset sits in it.
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
DELTAS = np.arange(-5.0, 5.01, 0.25)


def agree_frac(p_w, K, depth, R, t):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
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
        return 0.0, 0.0
    iu = np.clip(np.round(u[cand]).astype(int), 1, W - 2)
    iv = np.clip(np.round(v[cand]).astype(int), 1, H - 2)
    zm = depth[iv, iu].astype(float)
    good = np.isfinite(zm) & (zm > 0)
    rel = np.abs(zm - z[cand]) / np.maximum(z[cand], 1e-6)
    n_agree = int(np.sum(good & (rel < TAU)))
    return n_agree / len(p_w), n_agree / max(len(cand), 1)


def main():
    rows = []
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
        p_w = np.asarray(ref_c.as_o3d().points)
        err = M[:3, 3] - T[:3, 3]
        lat = float(np.linalg.norm(err[:2]))
        if lat < 1e-6:
            continue
        d = err[:2] / lat            # unit direction along which the estimate is wrong
        d3 = np.array([d[0], d[1], 0.0])

        prof = []
        for delta in DELTAS:
            t = T[:3, 3] + delta * d3
            fr, cond = agree_frac(p_w, mov_c.K, mov_c.depth, R_est, t)
            prof.append((float(delta), float(fr), float(cond)))
        # delta=0 is the TRUE pose; the estimate sits at delta = lat
        at_true = [p for p in prof if abs(p[0]) < 1e-9][0]
        at_est = min(prof, key=lambda p: abs(p[0] - lat))
        peak = max(p[1] for p in prof)
        width = float(sum(1 for p in prof if p[1] >= 0.9 * peak)) * 0.25
        rows.append({'scene': scene, 'tag': tag, 'sid': sid,
                     'lateral': lat, 'dir': d.tolist(),
                     'agree_at_true': at_true[1], 'agree_at_est': at_est[1],
                     'peak_agree': peak, 'width_90_at_0.25m': width,
                     'profile': prof})
        print('%-14s%-4s%-12s lat %5.2f m | agree@true %.3f  agree@est %.3f  peak %.3f  width90 %4.1f m'
              % (scene, tag, sid, lat, at_true[1], at_est[1], peak, width), flush=True)

    p = ROOT / 'out' / 'selfconsistency_width.json'
    p.write_text(json.dumps(rows, indent=1))
    print('\nn=%d' % len(rows))
    w = np.array([r['width_90_at_0.25m'] for r in rows])
    rt = np.array([r['agree_at_true'] for r in rows])
    re_ = np.array([r['agree_at_est'] for r in rows])
    print('width at 90%% of peak: median %.2f m  (min %.2f, max %.2f)'
          % (np.median(w), w.min(), w.max()))
    print('agreement at true pose: median %.3f   at estimated pose: median %.3f'
          % (np.median(rt), np.median(re_)))
    print('targets where the WRONG pose agrees at least as well as the true one: %d/%d'
          % (int((re_ >= rt).sum()), len(rows)))
    print('targets whose agreement profile stays wide (>=2 m): %d/%d'
          % (int((w >= 2.0).sum()), len(rows)))
    print('wrote', p)


if __name__ == '__main__':
    main()
