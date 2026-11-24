#!/usr/bin/env python
"""Multi-view registration workbench for the real-geometry study.

A task hands the agent several local view clouds (each in its OWN camera frame, built from
the registered laser scan) and asks it to bring every non-canonical view into the canonical
view's frame through one consistent set of transforms.  A pair solve places B where A
currently sits, so chains through intermediate views are possible.

No ground-truth pose is present in this process or in its data directory.

state: st['poses'][sid] = M mapping that view's own-frame points into the scene frame.
The canonical view has M = I; every other view also starts at M = I (i.e. unregistered).

Commands (all print JSON; an "images" field names PNGs to open):
  state                          matrices, budget, status                       (free)
  list                           member views                                   (free)
  photo <sid...>                 real RGB of these views                        (1)
  render [camera]                all placed clouds in the scene frame           (1)
  renderpair <a> <b> [camera]    only A and B, colour-coded                     (1)
  match <a> <b> [sift|orb|akaze] 2D features -> 3D correspondences              (1)
  solve <a> <b> rigid|similarity put B on A using stored correspondences        (1)
  solve_translation <a> <b>      B on A, translation only (rotation kept)       (1)
  icp <a> <b> [radius]           point-to-plane ICP of B onto A                 (1)
  fpfh <a> <b>                   FPFH+RANSAC+ICP of B onto A                    (1)
  link <a> <b>                   report which views are already connected       (free)
  undo                           revert the last state change                  (free)
  finish --status done|abstain --reason "..."
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent
SCENES = pathlib.Path(os.environ.get('MVRE_SCENES', 'E:/mvrecon_real/scenes'))
SESS = pathlib.Path(os.environ.get('MVRE_MV_SESSIONS', ROOT / 'mv_sessions'))
SCALE = 0.25
BUDGET = {'photo': 8, 'render': 20, 'match': 10, 'solve': 12, 'icp': 8, 'fpfh': 3,
          'total': 45}
RADIUS = 0.12


# ------------------------------------------------------------------ io
def san(x):
    if isinstance(x, pathlib.PurePath):
        return str(x)
    if isinstance(x, dict):
        return {str(k): san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [san(v) for v in x]
    if isinstance(x, np.ndarray):
        return san(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def emit(o):
    print(json.dumps(san(o), indent=1))


def sdir(task):
    return SESS / task


class Lock:
    def __init__(self, task, timeout=300.0):
        self.path = sdir(task) / '.lock'
        self.t = timeout

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() - t0 > self.t:
                    raise TimeoutError('busy')
                time.sleep(0.05)

    def __exit__(self, *e):
        try:
            os.unlink(self.path)
        except OSError:
            pass
        return False


def load(task):
    p = sdir(task) / 'state.json'
    if not p.exists():
        raise SystemExit('unknown task: %s' % task)
    return json.loads(p.read_text())


def save(st):
    (sdir(st['task']) / 'state.json').write_text(json.dumps(san(st), indent=1))


def trace(st, op, params, result=None, images=None, before=None, after=None, charged=True):
    st['step'] += 1
    rec = {'step': st['step'], 'op': op, 'params': san(params), 'charged': charged,
           'images': san(images or []), 'matrix_before': san(before),
           'matrix_after': san(after), 'result': san(result),
           'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(sdir(st['task']) / 'trace.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    save(st)
    return rec


def charge(st, kind, n=1):
    b, u = st['budget'], st['used']
    if u['total'] + n > b['total'] or u[kind] + n > b[kind]:
        return False
    u['total'] += n
    u[kind] += n
    return True


def no_budget(st, kind):
    return {'error': 'budget exhausted for %s' % kind, 'used': st['used'],
            'budget': st['budget']}


# ------------------------------------------------------------------ transforms
def M_of(st, sid):
    p = st['poses'][sid]
    M = np.eye(4)
    M[:3, :3] = np.asarray(p['R'], float) * float(p['s'])
    M[:3, 3] = np.asarray(p['t'], float)
    return M


def set_M(st, sid, M):
    s = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    st['poses'][sid] = {'R': (M[:3, :3] / s).tolist(), 't': M[:3, 3].tolist(), 's': s}


def push(st):
    st.setdefault('history', []).append({k: dict(v) for k, v in st['poses'].items()})


def all_M(st):
    return {s: M_of(st, s) for s in st['members']}


# ------------------------------------------------------------------ clouds
_VIEW = {}


def view(scene, sid):
    k = (scene, sid)
    if k not in _VIEW:
        d = np.load(SCENES / scene / (sid + '.npz'))
        img = cv2.cvtColor(cv2.imread(str(SCENES / scene / 'rgb' / str(d['rgb']))),
                           cv2.COLOR_BGR2RGB)
        _VIEW[k] = {'points': d['points'].astype(np.float64),
                    'colors': d['colors'].astype(np.float64),
                    'K': d['K'].astype(np.float64), 'W': int(d['W']), 'H': int(d['H']),
                    'rgb': img}
    return _VIEW[k]


def projected(v, P):
    """Image coordinates of own-frame points P in view v, and their pixel index."""
    K = v['K']
    z = np.clip(P[:, 2], 1e-6, None)
    u = K[0, 0] * P[:, 0] / z + K[0, 2]
    vv = K[1, 1] * P[:, 1] / z + K[1, 2]
    return u, vv


def placed(st, sid, P_own=None):
    """Own-frame points of view sid, expressed in the task's shared scene frame.

    M is the view's own placement inside the scene coordinate system chosen by the task;
    the scene frame itself is arbitrary, so this is exactly what the agent manipulates."""
    v = view(st['scene'], sid)
    P = v['points'] if P_own is None else P_own
    M = M_of(st, sid)
    return P @ M[:3, :3].T + M[:3, 3]


def thin(P, C, n):
    if len(P) <= n:
        return P, C
    i = np.random.default_rng(0).choice(len(P), n, replace=False)
    return P[i], C[i]


def pair_stats(st, a, b, radius=RADIUS):
    pa = placed(st, a)
    pb = placed(st, b)
    d, _ = cKDTree(pb).query(pa, k=1)
    d2, _ = cKDTree(pa).query(pb, k=1)
    return {'a_near_b': float((d < radius).mean()), 'b_near_a': float((d2 < radius).mean()),
            'median_nn_m': float(np.median(d)), 'n_a': int(len(pa)), 'n_b': int(len(pb))}


def kabsch(A, B, with_scale=False):
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = float(S.sum() / max(((A - ca) ** 2).sum(), 1e-12)) if with_scale else 1.0
    return s, R, cb - s * R @ ca


def ransac(Pm, Pr, with_scale, iters=2000, tol=0.08, seed=0):
    rng = np.random.default_rng(seed)
    n = len(Pm)
    if n < 6:
        return None
    best = (-1, None)
    for _ in range(iters):
        idx = rng.choice(n, 4 if with_scale else 3, replace=False)
        try:
            s, R, t = kabsch(Pm[idx], Pr[idx], with_scale)
        except np.linalg.LinAlgError:
            continue
        r = np.linalg.norm((Pm * s) @ R.T + t - Pr, axis=1)
        k = int((r < tol).sum())
        if k > best[0]:
            best = (k, np.where(r < tol)[0])
    if best[1] is None or len(best[1]) < 6:
        return None
    s, R, t = kabsch(Pm[best[1]], Pr[best[1]], with_scale)
    return s, R, t, int(len(best[1])), n


# ------------------------------------------------------------------ ops (observation)
def op_state(st):
    return {'task': st['task'], 'scene': st['scene'], 'kind': st['kind'],
            'canonical': st['canonical'], 'members': st['members'],
            'status': st['status'], 'step': st['step'],
            'scene_frame_matrices': all_M(st), 'used': st['used'],
            'budget': st['budget'], 'stored_correspondences': {
                k: v['n'] for k, v in st.get('corr', {}).items()}}


def op_list(st):
    out = []
    for s in st['members']:
        v = view(st['scene'], s)
        out.append({'sid': s, 'n_points': int(len(v['points'])), 'W': v['W'], 'H': v['H'],
                    'canonical': s == st['canonical']})
    return {'views': out}


def draw_grid(im, sc):
    H, W = im.shape[:2]
    for u in range(0, 6049, 500):
        x = int(round(u * sc))
        if x >= W:
            break
        cv2.line(im, (x, 0), (x, H - 1), (0, 255, 255), 1)
        cv2.putText(im, str(u), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 255), 1, cv2.LINE_AA)
    for v in range(0, 4033, 500):
        y = int(round(v * sc))
        if y >= H:
            break
        cv2.line(im, (0, y), (W - 1, y), (0, 255, 255), 1)
        cv2.putText(im, str(v), (2, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return im


def op_photo(st, sids):
    if not charge(st, 'photo'):
        return no_budget(st, 'photo')
    h = 430
    panels = []
    for s in sids:
        if s not in st['members']:
            return {'error': 'not a member view: %s' % s}
        v = view(st['scene'], s)
        sc = h / v['rgb'].shape[0]
        im = cv2.resize(v['rgb'], (int(v['rgb'].shape[1] * sc), h),
                        interpolation=cv2.INTER_AREA)
        im = draw_grid(im, sc)
        cv2.putText(im, ('CANON ' if s == st['canonical'] else '') + s, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(im)
        panels.append(np.full((h, 10, 3), 255, np.uint8))
    p = sdir(st['task']) / ('photos_%03d.png' % (st['step'] + 1))
    cv2.imwrite(str(p), cv2.cvtColor(np.hstack(panels[:-1]), cv2.COLOR_RGB2BGR))
    rec = trace(st, 'photo', {'sids': sids}, {'images': [str(p)]}, images=[str(p)])
    return {'images': [str(p)], 'step': rec['step'], 'used': st['used']}


CAMS = {'iso': (28, -55), 'top': (88, -90), 'side': (4, 0), 'front': (12, -90),
        'back': (12, 90), 'left': (12, 0)}


def _scatter_clouds(st, items, camera, size=7.4, title=''):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection='3d')
    allp = []
    for s, color, label in items:
        v = view(st['scene'], s)
        P = placed(st, s)
        P, C = thin(P, v['colors'], 4500)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1.4, c=(C if color is None else color),
                   label=label)
        allp.append(P)
    allp = np.vstack(allp)
    c = np.median(allp, 0)
    r = max(np.percentile(np.linalg.norm(allp - c, axis=1), 92), 0.5)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(*CAMS.get(camera, CAMS['iso']))
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title(title or ('scene frame, camera=%s' % camera))
    fig.tight_layout()
    return fig


def op_render(st, camera='iso'):
    if not charge(st, 'render'):
        return no_budget(st, 'render')
    items = [(s, None, ('CANON ' if s == st['canonical'] else '') + s) for s in st['members']]
    fig = _scatter_clouds(st, items, camera)
    p = sdir(st['task']) / ('render_%03d_%s.png' % (st['step'] + 1, camera))
    fig.savefig(p, bbox_inches='tight', dpi=110)
    plt.close(fig)
    rec = trace(st, 'render', {'camera': camera}, {'images': [str(p)]}, images=[str(p)])
    return {'images': [str(p)], 'step': rec['step'], 'used': st['used']}


def op_renderpair(st, a, b, camera='iso'):
    if not charge(st, 'render'):
        return no_budget(st, 'render')
    items = [(a, None, ('CANON ' if a == st['canonical'] else '') + a),
             (b, np.tile([0.85, 0.15, 0.1], (len(view(st['scene'], b)['points']), 1)),
              b + ' (red)')]
    fig = _scatter_clouds(st, items, camera)
    p = sdir(st['task']) / ('pair_%03d_%s_%s.png' % (st['step'] + 1, a, b))
    fig.savefig(p, bbox_inches='tight', dpi=110)
    plt.close(fig)
    rec = trace(st, 'renderpair', {'a': a, 'b': b}, {'images': [str(p)]},
                images=[str(p)])
    return {'images': [str(p)], 'step': rec['step'], 'used': st['used'],
            'pair_stats': pair_stats(st, a, b)}


# ------------------------------------------------------------------ ops (matching/solving)
def _detect(scene, sid, matcher):
    v = view(scene, sid)
    g = cv2.cvtColor(v['rgb'], cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    if matcher == 'sift':
        det, norm = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02), cv2.NORM_L2
    elif matcher == 'orb':
        det, norm = cv2.ORB_create(nfeatures=4000, fastThreshold=12), cv2.NORM_HAMMING
    elif matcher == 'akaze':
        det, norm = cv2.AKAZE_create(threshold=0.0008), cv2.NORM_HAMMING
    else:
        raise ValueError('matcher must be sift, orb or akaze')
    return det.detectAndCompute(g, None) + (norm,)


def _lift(v, uv_full, pixels_of_points):
    """Nearest visible scan point (own-frame) to each 2D keypoint."""
    u, vv = pixels_of_points
    pts = np.stack([u, vv], 1)
    uvc = np.array([uv_full[:, 0], uv_full[:, 1]]).T
    d, idx = cKDTree(pts).query(uvc, k=1)
    ok = d < 6.0
    return idx, ok, d


def op_match(st, a, b, matcher='sift'):
    if not charge(st, 'match'):
        return no_budget(st, 'match')
    if a not in st['members'] or b not in st['members'] or a == b:
        return {'error': 'a and b must be distinct member views'}
    k1, d1, norm = _detect(st['scene'], a, matcher)
    k2, d2, _ = _detect(st['scene'], b, matcher)
    if d1 is None or d2 is None:
        return {'error': 'no descriptors'}
    bf = cv2.BFMatcher(norm)
    m = bf.knnMatch(d1, d2, k=2)
    good = [x for x, y in m if x.distance < 0.8 * y.distance]
    m2 = bf.knnMatch(d2, d1, k=2)
    back = {x.queryIdx: x.trainIdx for x, y in m2 if x.distance < 0.8 * y.distance}
    good = [x for x in good if back.get(x.trainIdx) == x.queryIdx]
    if len(good) < 8:
        trace(st, 'match', {'a': a, 'b': b}, {'error': 'too few', 'n': len(good)})
        return {'error': 'too few matches', 'n': len(good), 'used': st['used']}
    ua = np.array([k1[x.queryIdx].pt for x in good], float) / SCALE
    ub = np.array([k2[x.trainIdx].pt for x in good], float) / SCALE
    va, vb = view(st['scene'], a), view(st['scene'], b)
    Pa_own, Pb_own = va['points'], vb['points']
    idx_a, ok_a, da = _lift(va, ua, projected(va, Pa_own))
    idx_b, ok_b, db = _lift(vb, ub, projected(vb, Pb_own))
    keep = ok_a & ok_b
    A3 = Pa_own[idx_a[keep]]
    B3 = Pb_own[idx_b[keep]]
    st.setdefault('corr', {})['%s|%s|%s' % (a, b, matcher)] = {
        'a': a, 'b': b, 'matcher': matcher, 'n': int(keep.sum())}
    np.savez_compressed(sdir(st['task']) / ('corr_%s_%s_%s.npz' % (a, b, matcher)),
                        A=A3, B=B3)
    # Visualisation of the 2D matches
    h = 360
    sca = h / va['rgb'].shape[0]
    ia = cv2.resize(va['rgb'], (int(va['rgb'].shape[1] * sca), h), interpolation=cv2.INTER_AREA)
    ib = cv2.resize(vb['rgb'], (int(vb['rgb'].shape[1] * sca), h), interpolation=cv2.INTER_AREA)
    gap = np.full((h, 8, 3), 255, np.uint8)
    out = np.hstack([ia, gap, ib])
    off = ia.shape[1] + 8
    rng = np.random.default_rng(1)
    sel = np.arange(len(ua))
    if len(sel) > 60:
        sel = rng.choice(sel, 60, replace=False)
    for i in sel:
        if not keep[i]:
            continue
        p1 = (int(ua[i, 0] * sca), int(ua[i, 1] * sca))
        p2 = (int(ub[i, 0] * sca) + off, int(ub[i, 1] * sca))
        col = (0, 220, 0) if i % 2 else (255, 150, 0)
        cv2.circle(out, p1, 3, col, -1)
        cv2.circle(out, p2, 3, col, -1)
        cv2.line(out, p1, p2, col, 1)
    p = sdir(st['task']) / ('match_%03d_%s_%s_%s.png' % (st['step'] + 1, a, b, matcher))
    cv2.imwrite(str(p), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    res = {'matches_2d': len(good), 'matches_3d': int(keep.sum()),
           'median_lift_err_px': float(np.median(np.maximum(da[ok_a], db[ok_b])))
           if keep.any() else None, 'images': [str(p)],
           'key': '%s|%s|%s' % (a, b, matcher)}
    rec = trace(st, 'match', {'a': a, 'b': b, 'matcher': matcher}, res, images=[str(p)])
    res['step'] = rec['step']
    res['used'] = st['used']
    return res


def _corr(st, a, b, matcher):
    f = sdir(st['task']) / ('corr_%s_%s_%s.npz' % (a, b, matcher))
    if not f.exists():
        return None
    d = np.load(f)
    return d['A'], d['B']


def op_solve(st, a, b, mode='rigid'):
    if not charge(st, 'solve'):
        return no_budget(st, 'solve')
    best = None
    for matcher in ('sift', 'akaze', 'orb'):
        c = _corr(st, a, b, matcher)
        if c is None or len(c[0]) < 8:
            continue
        if best is None or len(c[0]) > len(best[0]):
            best = c
    if best is None:
        trace(st, 'solve', {'a': a, 'b': b, 'mode': mode}, {'error': 'no correspondences'})
        return {'error': 'no stored correspondences for this pair; run match first',
                'used': st['used']}
    A3, B3 = best                       # a's and b's own-frame 3D points
    Ma = M_of(st, a)
    Pr = A3 @ Ma[:3, :3].T + Ma[:3, 3]  # target positions in the scene frame
    Pm = B3                             # b's points, own frame
    before = M_of(st, b)
    info = {'a': a, 'b': b, 'mode': mode, 'n_corr': int(len(A3))}
    if mode == 'translation':
        R = np.linalg.inv(before[:3, :3])
        t = np.median(Pr - Pm @ R.T, axis=0)
        Mnew = np.eye(4)
        Mnew[:3, :3] = R
        Mnew[:3, 3] = t
    else:
        fit = ransac(Pm, Pr, with_scale=(mode == 'similarity'))
        if fit is None:
            trace(st, 'solve', {'a': a, 'b': b, 'mode': mode},
                  {'error': 'RANSAC no consensus', 'n': len(A3)})
            return {'error': 'RANSAC found no consensus', 'n': len(A3),
                    'used': st['used']}
        sfit, Rfit, tfit, inl, n = fit
        Mnew = np.eye(4)
        Mnew[:3, :3] = Rfit * sfit
        Mnew[:3, 3] = tfit
        info.update(inliers=inl, n=n, scale=sfit if mode == 'similarity' else 1.0)
    push(st)
    set_M(st, b, Mnew)
    stt = pair_stats(st, a, b)
    res = {**info, **stt}
    rec = trace(st, 'solve', {'a': a, 'b': b, 'mode': mode}, res,
                before=before, after=Mnew)
    res['step'] = rec['step']
    res['used'] = st['used']
    res['scene_frame_matrices'] = all_M(st)
    return res


def _o3d_cloud(P, C):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    return pc


def op_icp(st, a, b, radius=0.25):
    if not charge(st, 'icp'):
        return no_budget(st, 'icp')
    import open3d as o3d
    va, vb = view(st['scene'], a), view(st['scene'], b)
    src = _o3d_cloud(placed(st, b), vb['colors'])
    dst = _o3d_cloud(placed(st, a), va['colors'])
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=24))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=24))
    r = o3d.pipelines.registration.registration_icp(
        src, dst, radius, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    before = M_of(st, b)
    Mnew = r.transformation @ before
    push(st)
    set_M(st, b, Mnew)
    stt = pair_stats(st, a, b)
    res = {'placed': b, 'on': a, 'fitness': float(r.fitness),
           'rmse': float(r.inlier_rmse), **stt}
    rec = trace(st, 'icp', {'a': a, 'b': b, 'radius': radius}, res,
                before=before, after=Mnew)
    res['step'] = rec['step']
    res['used'] = st['used']
    return res


def op_fpfh(st, a, b):
    if not charge(st, 'fpfh'):
        return no_budget(st, 'fpfh')
    import open3d as o3d
    o3d.utility.random.seed(0)
    va, vb = view(st['scene'], a), view(st['scene'], b)
    src = _o3d_cloud(placed(st, b), vb['colors'])
    dst = _o3d_cloud(placed(st, a), va['colors'])
    fs = src.voxel_down_sample(0.4)
    ft = dst.voxel_down_sample(0.4)
    fs.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    ft.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    Fs = o3d.pipelines.registration.compute_fpfh_feature(
        fs, o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=60))
    Ft = o3d.pipelines.registration.compute_fpfh_feature(
        ft, o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=60))
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        fs, ft, Fs, Ft, True, 0.3,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [],
        o3d.pipelines.registration.RANSACConvergenceCriteria(60000, 0.95))
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    fine = o3d.pipelines.registration.registration_icp(
        src, dst, 0.25, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
    before = M_of(st, b)
    Mnew = fine.transformation @ before
    push(st)
    set_M(st, b, Mnew)
    stt = pair_stats(st, a, b)
    res = {'placed': b, 'on': a, 'ransac_fitness': float(coarse.fitness),
           'icp_fitness': float(fine.fitness), 'icp_rmse': float(fine.inlier_rmse), **stt}
    rec = trace(st, 'fpfh', {'a': a, 'b': b}, res, before=before, after=Mnew)
    res['step'] = rec['step']
    res['used'] = st['used']
    return res


def op_link(st):
    """Which member views are already mutually consistent (pair stats in both directions)."""
    out = {}
    for a in st['members']:
        for b in st['members']:
            if a < b:
                out['%s|%s' % (a, b)] = pair_stats(st, a, b)
    return {'pairs': out}


def op_undo(st):
    h = st.get('history') or []
    if not h:
        return {'ok': False, 'error': 'nothing to undo'}
    st['poses'] = h.pop()
    rec = trace(st, 'undo', {}, {'ok': True}, charged=False)
    return {'ok': True, 'step': rec['step'], 'scene_frame_matrices': all_M(st)}


def op_finish(st, status, reason):
    if status not in ('done', 'abstain'):
        return {'error': 'status must be done or abstain'}
    st['status'] = status
    st['reason'] = reason
    rec = trace(st, 'finish', {'status': status, 'reason': reason},
                {'scene_frame_matrices': all_M(st)}, charged=False)
    return {'status': status, 'reason': reason, 'step': rec['step'],
            'scene_frame_matrices': all_M(st)}


# ------------------------------------------------------------------ CLI
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('task')
    ap.add_argument('command')
    ap.add_argument('rest', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv[1:])
    try:
        with Lock(a.task):
            st = load(a.task)
            if st['status'] != 'running' and a.command not in ('state', 'list'):
                emit({'error': 'session finished', 'status': st['status']})
                return 0
            return dispatch(st, a.command, a.rest)
    except Exception as e:
        import traceback
        emit({'error': '%s: %s' % (type(e).__name__, e),
              'traceback_tail': traceback.format_exc().splitlines()[-3:]})
        return 1


def dispatch(st, cmd, rest):
    if cmd == 'state':
        emit(op_state(st)); return 0
    if cmd == 'list':
        emit(op_list(st)); return 0
    if cmd == 'link':
        emit(op_link(st)); return 0
    if cmd == 'photo':
        emit(op_photo(st, rest or st['members'])); return 0
    if cmd == 'render':
        emit(op_render(st, rest[0] if rest else 'iso')); return 0
    if cmd == 'renderpair':
        emit(op_renderpair(st, rest[0], rest[1], rest[2] if len(rest) > 2 else 'iso'))
        return 0
    if cmd == 'match':
        emit(op_match(st, rest[0], rest[1], rest[2] if len(rest) > 2 else 'sift')); return 0
    if cmd == 'solve':
        emit(op_solve(st, rest[0], rest[1], rest[2] if len(rest) > 2 else 'rigid')); return 0
    if cmd == 'solve_translation':
        emit(op_solve(st, rest[0], rest[1], 'translation')); return 0
    if cmd == 'icp':
        emit(op_icp(st, rest[0], rest[1], float(rest[2]) if len(rest) > 2 else 0.25))
        return 0
    if cmd == 'fpfh':
        emit(op_fpfh(st, rest[0], rest[1])); return 0
    if cmd == 'undo':
        emit(op_undo(st)); return 0
    if cmd == 'finish':
        status, reason = 'done', ''
        i = 0
        while i < len(rest):
            if rest[i] == '--status':
                status = rest[i + 1]; i += 2
            elif rest[i] == '--reason':
                reason = ' '.join(rest[i + 1:]); break
            else:
                i += 1
        emit(op_finish(st, status, reason)); return 0
    emit({'error': 'unknown command', 'commands': [
        'state', 'list', 'link', 'photo', 'render', 'renderpair', 'match', 'solve',
        'solve_translation', 'icp', 'fpfh', 'undo', 'finish']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
