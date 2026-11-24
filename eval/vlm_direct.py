"""Let the large model do the registration directly, blind, and score only what it said.

The previous loop had a model choosing actions, but the transform came from SIFT correspondences
and a least-squares fit. This is the other thing: the model's own output IS the registration.

Protocol, enforced by the tool rather than by good intentions:

  crops   produce zoomed panels of both photographs with fine pixel grids
  answer  the model writes its correspondences -- named objects and their pixel positions in each
          image -- to a file. Nothing is scored at this point.
  score   REFUSES to run unless the answer file already exists. Computes the transform from the
          model's pairs alone (Kabsch; translation is exactly determined once rotation is), and
          reports the error.

Because the transform is computed only from the named pairs, the model cannot smuggle a rotation
into it: every contribution is a statement of the form "this object at (u,v) here is that object at
(u,v) there", which anyone can check against the crops. That is what makes this auditable despite
the fact that I have already seen ground-truth-derived diagnostics for some of these targets.

  crops  <scene> <tag> <sid>
  answer <scene> <tag> <sid> <json>     json: {"pairs":[[ur,vr,um,vm,"what it is"],...],
                                              "pose_guess_deg":[rx,ry,rz]}
  score  <scene> <tag> <sid>
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
from protocol import setup  # noqa: E402
from wb import Workbench  # noqa: E402

TASK = json.loads((ROOT / 'eval_data' / 'task.json').read_text())
BASE = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
ANS = ROOT / 'out' / 'vlm_direct'


def _ctx(scene, tag, sid):
    wb = Workbench(ROOT / 'agent_data' / scene, stride=32)
    setup(wb, TASK[scene][tag], TASK[scene][tag]['ref'])
    x = next(r for r in BASE if r['method'] == 'A_icp' and r['scene'] == scene
             and r['tag'] == tag and r['sid'] == sid)
    return wb, x


def _panel(img, u0, u1, v0, v1, w=760, g=50):
    sub = img[v0:v1, u0:u1].copy()
    f = w / sub.shape[1]
    sm = cv2.resize(sub, (w, int(round(sub.shape[0] * f))), interpolation=cv2.INTER_AREA)
    for u in range(u0 - u0 % g, u1, g):
        x = int((u - u0) * f)
        if 0 <= x < sm.shape[1]:
            cv2.line(sm, (x, 0), (x, sm.shape[0]), (0, 255, 0), 1)
            if u % 100 == 0:
                cv2.putText(sm, str(u), (x + 1, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            (0, 255, 0), 1)
    for v in range(v0 - v0 % g, v1, g):
        y = int((v - v0) * f)
        if 0 <= y < sm.shape[0]:
            cv2.line(sm, (0, y), (sm.shape[1], y), (0, 255, 0), 1)
            if v % 100 == 0:
                cv2.putText(sm, str(v), (1, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            (0, 255, 0), 1)
    return sm


def crops(scene, tag, sid):
    wb, _ = _ctx(scene, tag, sid)
    out = []
    for s in (wb.ref, sid):
        img = cv2.imread(str(ROOT / 'agent_data' / scene / (s + '.png')))
        H, W = img.shape[:2]
        left = _panel(img, 0, W // 2, 0, H)
        right = _panel(img, W // 2, W, 0, H)
        h = max(left.shape[0], right.shape[0])
        left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        row = np.hstack([left, right])
        cv2.putText(row, '%s  %dx%d  grid=50px full-res' % (s, W, H),
                    (6, row.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        out.append(row)
    w = min(r.shape[1] for r in out)
    sheet = np.vstack([r[:, :w] for r in out])
    p = ANS / ('%s_%s_%s_crops.jpg' % (scene, tag, sid))
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(p)
    print('reference=%s  moving=%s' % (wb.ref, sid))
    return p


def zoom(scene, tag, sid, ur0, ur1, vr0, vr1, um0, um1, vm0, vm1):
    wb, _ = _ctx(scene, tag, sid)
    rows = []
    for s, (u0, u1, v0, v1) in ((wb.ref, (ur0, ur1, vr0, vr1)),
                                (sid, (um0, um1, vm0, vm1))):
        img = cv2.imread(str(ROOT / 'agent_data' / scene / (s + '.png')))
        pnl = _panel(img, int(u0), int(u1), int(v0), int(v1), w=1100, g=25)
        cv2.putText(pnl, s, (6, pnl.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        rows.append(pnl)
    w = min(r.shape[1] for r in rows)
    sheet = np.vstack([r[:, :w] for r in rows])
    p = ANS / ('%s_%s_%s_zoom.jpg' % (scene, tag, sid))
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(p)
    return p


def mark(scene, tag, sid, ur, vr, um, vm, half=110):
    wb, _ = _ctx(scene, tag, sid)
    rows = []
    for s, (u, v) in ((wb.ref, (ur, vr)), (sid, (um, vm))):
        img = cv2.imread(str(ROOT / 'agent_data' / scene / (s + '.png')))
        u0, u1 = int(u - half), int(u + half)
        v0, v1 = int(v - half), int(v + half)
        pnl = _panel(img, u0, u1, v0, v1, w=520, g=25)
        cx = int((u - u0) * (520 / (u1 - u0)))
        cy = int((v - v0) * (520 / (u1 - u0)))
        cv2.drawMarker(pnl, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.putText(pnl, '%s (%.0f,%.0f)' % (s, u, v), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rows.append(pnl)
    h = max(r.shape[0] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT) for r in rows]
    out = np.hstack(rows)
    p = ANS / ('%s_%s_%s_mark.jpg' % (scene, tag, sid))
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(p)
    return p


def answer(scene, tag, sid, payload):
    p = ANS / ('%s_%s_%s_answer.json' % (scene, tag, sid))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1))
    print(p)
    print('committed %d pairs; now run score' % len(payload.get('pairs', [])))
    return p


def _bp(u, v, c):
    H, W = c.depth.shape
    iu, iv = int(round(u)), int(round(v))
    if not (2 <= iu < W - 2 and 2 <= iv < H - 2):
        return None
    z = float(np.median(c.depth[iv - 1:iv + 2, iu - 1:iu + 2]))
    if not np.isfinite(z) or z <= 0:
        return None
    fx, fy, cx, cy = c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2]
    return np.array([(iu - cx) / fx * z, (iv - cy) / fy * z, z])


def score(scene, tag, sid):
    p = ANS / ('%s_%s_%s_answer.json' % (scene, tag, sid))
    if not p.exists():
        print('REFUSED: no committed answer at %s -- write the answer first' % p)
        return 2
    ans = json.loads(p.read_text())
    wb, x = _ctx(scene, tag, sid)
    ref_c, mov_c = wb.clouds[wb.ref], wb.clouds[sid]
    T = np.asarray(x['matrix_gt'], float)
    R_true, t_true = T[:3, :3], T[:3, 3]

    Pr, Pm, used = [], [], []
    for row in ans['pairs']:
        ur, vr, um, vm = row[:4]
        pr, pm = _bp(ur, vr, ref_c), _bp(um, vm, mov_c)
        if pr is None or pm is None:
            print('  pair %s: depth invalid, skipped' % (row[4:] or row[:4]))
            continue
        Pr.append(pr)
        Pm.append(pm)
        used.append(row)
    if len(Pr) < 3:
        print('REFUSED: %d usable pair(s). Two pairs cannot determine a rotation -- the centred '
              'cross-covariance has rank 1, so the rotation about the axis joining them is free '
              'and SVD returns an arbitrary solution. At least 3 non-collinear pairs are needed.'
              % len(Pr))
        return 2
    Pr, Pm = np.array(Pr), np.array(Pm)
    ca, cb = Pm.mean(0), Pr.mean(0)
    H = (Pm - ca).T @ (Pr - cb)
    rank = int(np.linalg.matrix_rank(H))
    if rank < 2:
        print('REFUSED: the pairs are collinear (rank %d) -- rotation is underdetermined.' % rank)
        return 2
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R @ R_true.T) - 1) / 2, -1, 1))))
    lat = float(np.linalg.norm((t - t_true)[:2]))
    full = float(np.linalg.norm(t - t_true))
    out = {'n_pairs_used': len(used), 'rank_H': rank, 'pairs': used,
           'rot_err_deg': rot, 'lateral_err_m': lat, 'trans_err_m': full,
           'baseline_rot_deg': x['score_corrected']['rot_err_deg'],
           'baseline_trans_m': x['score_corrected']['trans_err_m']}
    (ANS / ('%s_%s_%s_score.json' % (scene, tag, sid))).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('n_pairs_used', 'rot_err_deg', 'lateral_err_m',
                                         'trans_err_m', 'baseline_rot_deg',
                                         'baseline_trans_m')}))
    return 0


def main(argv):
    if len(argv) < 5:
        print(__doc__)
        return 2
    cmd, scene, tag, sid = argv[1:5]
    if cmd == 'crops':
        crops(scene, tag, sid)
        return 0
    if cmd == 'zoom':
        v = [float(t) for t in argv[5:13]]
        zoom(scene, tag, sid, *v)
        return 0
    if cmd == 'mark':
        v = [float(t) for t in argv[5:9]]
        mark(scene, tag, sid, *v)
        return 0
    if cmd == 'answer':
        answer(scene, tag, sid, json.loads(argv[5]))
        return 0
    if cmd == 'score':
        return score(scene, tag, sid)
    print(__doc__)
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
