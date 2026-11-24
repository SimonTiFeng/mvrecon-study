"""Can the co-visibility anchor filter run WITHOUT ground truth, as an agent could?

anchor_visibility.py showed that ONE anchor on a surface both views genuinely observe takes
median lateral error 0.94 -> 0.13 m (20/25 under 0.5 m). But it defined "observes" using
T_true. An agent has only its current pose estimate, so this repeats the filter with the
ESTIMATED transform:

  1. project reference points through the current pose estimate into the moving view
  2. keep those landing in-frame whose predicted depth agrees with the moving view's own
     depth model within tau (this is the co-visibility test -- no GT)
  3. pick an anchor from that set, take the pixel the estimate says it lands on,
     backproject with the moving depth, solve t = p_ref - R p2   (R held at the baseline)
  4. repeat: the new t re-projects, re-selects, re-solves

If this converges to the oracle number, the method is implementable and the remaining gap is
a perception problem, not an information problem. If it does not, a mistaken pose sends the
anchor to the wrong pixel and the filter cannot rescue itself -- which is what the workbench's
render/undo loop would have to supply.

Also reports the REFUSAL case: rounds where no point survives the co-visibility test. That is
the "insufficient evidence" state, and it must be distinguishable from a bad solve.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
RNG = np.random.default_rng(3)


def flatness(pts, k=14):
    if len(pts) < k + 1:
        return np.full(len(pts), np.inf)
    _, idx = cKDTree(pts).query(pts, k=k)
    P = pts[idx] - pts[idx].mean(1, keepdims=True)
    C = np.einsum('nki,nkj->nij', P, P) / k
    return np.sqrt(np.clip(np.linalg.eigvalsh(C)[:, 0], 0.0, None))


def run(p_w, K, depth, R, t0, t_true, tau, k_anchor, rounds, rule, rng):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    t = t0.copy()
    hist, refused = [], []
    for _ in range(rounds):
        p_m = (p_w - t) @ R
        z = p_m[:, 2]
        ok_z = z > 0.2
        u = np.full(len(p_w), np.nan)
        v = np.full(len(p_w), np.nan)
        u[ok_z] = fx * p_m[ok_z, 0] / z[ok_z] + cx
        v[ok_z] = fy * p_m[ok_z, 1] / z[ok_z] + cy
        frame = ok_z & (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
        cand = np.where(frame)[0]
        if len(cand) == 0:
            refused.append('no_inframe'); break
        iu = np.clip(np.round(u[cand]).astype(int), 1, W - 2)
        iv = np.clip(np.round(v[cand]).astype(int), 1, H - 2)
        zm = depth[iv, iu].astype(float)
        good = np.isfinite(zm) & (zm > 0)
        rel = np.abs(zm - z[cand]) / np.maximum(z[cand], 1e-6)
        survive = cand[good & (rel < tau)]
        if len(survive) == 0:
            refused.append('no_covisible'); break
        if rule == 'flat':
            f = flatness(p_w[survive])
            order = np.argsort(f)
            pick = survive[order[:k_anchor]] if k_anchor > 1 else survive[[int(np.argmin(f))]]
        else:
            k = min(k_anchor, len(survive))
            pick = survive[rng.choice(len(survive), k, replace=False)]
        fx_, fy_ = K[0, 0], K[1, 1]
        ts = []
        for i in pick:
            uu, vv = int(round(u[i])), int(round(v[i]))
            zz = float(np.median(depth[vv - 1:vv + 2, uu - 1:uu + 2]))
            if not np.isfinite(zz) or zz <= 0:
                continue
            p2 = np.array([(uu - cx) / fx_ * zz, (vv - cy) / fy_ * zz, zz])
            ts.append(p_w[i] - R @ p2)
        if not ts:
            refused.append('no_valid_backproject'); break
        t = np.median(np.array(ts), axis=0)
        hist.append(float(np.linalg.norm((t - t_true)[:2])))
    return t, hist, refused, len(survive) if 'survive' in dir() else 0


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
        row = {'scene': scene, 'tag': tag, 'sid': sid,
               'before': float(np.linalg.norm((M[:3, 3] - T[:3, 3])[:2]))}
        for tau in (0.05, 0.10, 0.20):
            for k in (1, 16):
                for rule in ('flat', 'random'):
                    t, hist, refused, _ = run(p_w, mov_c.K, mov_c.depth, R_est,
                                              M[:3, 3].copy(), T[:3, 3], tau, k, 4,
                                              rule, np.random.default_rng(5))
                    row['tau%.2f_k%d_%s' % (tau, k, rule)] = {
                        'final': float(np.linalg.norm((t - T[:3, 3])[:2])),
                        'hist': hist, 'refused': refused}
        rows.append(row)
        b = row['before']
        cells = ' '.join('%s:%.2f' % (key.split('_')[-2] + key.split('_')[-1],
                                     row[key]['final'])
                         for key in row if isinstance(row[key], dict))
        print('%-14s%-4s%-12s before %6.2f | %s' % (scene, tag, sid, b, cells), flush=True)

    p = ROOT / 'out' / 'anchor_est_loop.json'
    p.write_text(json.dumps(rows, indent=1))
    keys = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    print('\nn=%d   (before: median %.2f m, under 0.5 m %d/%d)'
          % (len(rows), np.median([r['before'] for r in rows]),
             int(sum(r['before'] < 0.5 for r in rows)), len(rows)))
    print('%-18s %8s %8s %8s %8s' % ('policy', 'median', '<0.5m', 'refused', 'med.rounds'))
    for key in keys:
        v = np.array([r[key]['final'] for r in rows])
        ref = sum(1 for r in rows if r[key]['refused'])
        rl = np.median([len(r[key]['hist']) for r in rows])
        print('%-18s %7.2fm %5d/%d %7d/%d %10.1f'
              % (key, np.median(v), int((v < 0.5).sum()), len(v), ref, len(rows), rl))
    print('wrote', p)


if __name__ == '__main__':
    main()
