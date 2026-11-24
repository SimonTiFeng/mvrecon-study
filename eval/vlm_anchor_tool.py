"""Supply an identity anchor from the images, and score what it buys.

This is the minimal version of the proposed method's core primitive, with a real vision-language
model in the loop (the agent reads the contact sheet and names a correspondence). Everything is
the agent-visible state only: the two original photographs and each view's own predicted depth.
No ground truth enters the choice.

  python eval/vlm_anchor_tool.py sheet <scene> <tag> <sid>
      writes out/sheets/<scene>_<tag>_<sid>.jpg -- the reference and moving photograph side by
      side, each with a labelled pixel grid in FULL-RESOLUTION coordinates, so a correspondence
      can be stated as two pixel positions.

  python eval/vlm_anchor_tool.py solve <scene> <tag> <sid> u_ref v_ref u_mov v_mov [...]
      backprojects each named pixel with that view's own depth, solves t = p_ref - R p_est, and
      reports the error against the reference transform (evaluation side only, for scoring).

Several correspondence pairs may be given; the translation is then the median of the individual
solves, which is how the agent would use more than one landmark.
"""
from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'workbench'))
from protocol import setup, rotdeg  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
SHEET_W = 1400          # per-panel width in the contact sheet
GRID = 250              # grid spacing in full-resolution pixels


def _load(scene, tag, sid):
    wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
    setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
    x = next(r for r in BASE
             if r['method'] == 'A_icp' and r['scene'] == scene
             and r['tag'] == tag and r['sid'] == sid)
    M = np.asarray(x['matrix_est'], float)
    R_est = M[:3, :3] / max(np.cbrt(abs(np.linalg.det(M[:3, :3]))), 1e-12)
    return wb, x, M, R_est


def sheet(scene, tag, sid):
    wb, x, M, R_est = _load(scene, tag, sid)
    ref_sid = wb.ref
    panels = []
    for s in (ref_sid, sid):
        im = cv2.imread(str(ROOT / 'agent_data' / scene / (s + '.png')))
        H, W = im.shape[:2]
        f = SHEET_W / W
        sm = cv2.resize(im, (SHEET_W, int(round(H * f))), interpolation=cv2.INTER_AREA)
        for u in range(0, W, GRID):
            cv2.line(sm, (int(u * f), 0), (int(u * f), sm.shape[0]), (0, 255, 0), 1)
            cv2.putText(sm, str(u), (int(u * f) + 3, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1)
        for v in range(0, H, GRID):
            cv2.line(sm, (0, int(v * f)), (sm.shape[1], int(v * f)), (0, 255, 0), 1)
            cv2.putText(sm, str(v), (3, int(v * f) + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1)
        cv2.putText(sm, '%s  (%dx%d)  GRID=%dpx full-res' % (s, W, H, GRID),
                    (8, sm.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panels.append(sm)
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0)) for p in panels]
    out = np.hstack(panels)
    p = ROOT / 'out' / 'sheets'
    p.mkdir(parents=True, exist_ok=True)
    f = p / ('%s_%s_%s.jpg' % (scene, tag, sid))
    cv2.imwrite(str(f), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f)
    print('reference view: %s   moving view: %s' % (ref_sid, sid))
    print('baseline translation error: %.2f m' % x['score_corrected']['trans_err_m'])
    return f


def solve(scene, tag, sid, pairs):
    wb, x, M, R_est = _load(scene, tag, sid)
    ref_sid = wb.ref
    ref_c, mov_c = wb.clouds[ref_sid], wb.clouds[sid]
    T = np.asarray(x['matrix_gt'], float)

    def bp(u, v, c):
        H, W = c.depth.shape
        iu, iv = int(round(u)), int(round(v))
        if not (2 <= iu < W - 2 and 2 <= iv < H - 2):
            return None
        z = float(np.median(c.depth[iv - 1:iv + 2, iu - 1:iu + 2]))
        if not np.isfinite(z) or z <= 0:
            return None
        fx, fy, cx, cy = c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2]
        return np.array([(iu - cx) / fx * z, (iv - cy) / fy * z, z])

    ts = []
    pairs3d = []
    for k in range(0, len(pairs), 4):
        ur, vr, um, vm = pairs[k:k + 4]
        pr = bp(ur, vr, ref_c)
        pm = bp(um, vm, mov_c)
        if pr is None or pm is None:
            print('pair %d: depth invalid at one of the pixels, skipped' % (k // 4 + 1))
            continue
        t = pr - R_est @ pm
        ts.append(t)
        pairs3d.append((pr, pm))
        print('pair %d: ref(%.0f,%.0f) z=%.2f  mov(%.0f,%.0f) z=%.2f  -> t=%s'
              % (k // 4 + 1, ur, vr, pr[2], um, vm, pm[2], np.round(t, 2)))
    if not ts:
        print('no usable pair')
        return
    t = np.median(np.array(ts), axis=0)
    lat = float(np.linalg.norm((t - T[:3, 3])[:2]))
    print('\nrigid (median of pairs): t = %s' % np.round(t, 3))
    print('lateral error %.2f m (baseline %.2f m)'
          % (lat, x['score_corrected']['trans_err_m']))

    # Similarity fit over the same anchors. With R held, two correspondences determine a
    # uniform scale as well as the translation, which absorbs a per-view depth bias.
    sim = None
    if len(ts) >= 2:
        Pr = np.array([a for a, _ in pairs3d])
        Pm = np.array([b for _, b in pairs3d])
        ca, cb = Pm.mean(0), Pr.mean(0)
        s_est = float(np.median(np.linalg.norm(Pr - cb, axis=1)
                                / np.maximum(np.linalg.norm(Pm - ca, axis=1), 1e-9)))
        t_sim = cb - s_est * (R_est @ ca)
        lat_sim = float(np.linalg.norm((t_sim - T[:3, 3])[:2]))
        sim = {'s': s_est, 't': t_sim.tolist(), 'lateral': lat_sim}
        print('similarity (%d anchors, s=%.3f): t = %s  -> lateral %.2f m'
              % (len(ts), s_est, np.round(t_sim, 3), lat_sim))
    print('true   t = %s' % np.round(T[:3, 3], 3))
    f_prev = ROOT / 'out' / 'vlm_anchors.json'
    res = json.loads(f_prev.read_text()) if f_prev.exists() else {}
    key = scene + '|' + tag + '|' + sid
    res[key] = {'pairs': [list(map(float, pairs[i:i + 4]))
                          for i in range(0, len(pairs), 4)],
                't': t.tolist(), 'lateral': lat, 'similarity': sim,
                'baseline_trans_err_m': x['score_corrected']['trans_err_m'],
                'rot_err': x['score_corrected']['rot_err_deg']}
    f_prev.write_text(json.dumps(res, indent=1))
    print('appended to out/vlm_anchors.json')


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(__doc__)
        raise SystemExit(2)
    mode, scene, tag, sid = sys.argv[1:5]
    if mode == 'sheet':
        sheet(scene, tag, sid)
    elif mode == 'solve':
        vals = [float(v) for v in sys.argv[5:]]
        if len(vals) < 4 or len(vals) % 4:
            print('need 4 numbers per correspondence (u_ref v_ref u_mov v_mov)')
            raise SystemExit(2)
        solve(scene, tag, sid, vals)
    else:
        print(__doc__)
        raise SystemExit(2)
