#!/usr/bin/env python
"""Agent-facing 3D registration workbench.

One process per command.  Session state, budget, undo stack and an audit trace live
under sessions/<case>/.  This file contains NO ground-truth pose and imports nothing
from the evaluation side.

Conventions
-----------
* The reference cloud is fixed.  The moving cloud has a cumulative similarity
  M = [s R | t] mapping its OWN camera-frame points into the reference frame.
* Every operation is an incremental similarity Q left-composed onto the current
  state: M_new = Q @ M_old.
* The initial cumulative transform of the moving cloud is whatever the harness
  wrote into session.json at creation (a deliberate displacement).
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

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'data'
SESS = ROOT / 'sessions'
SCALE = 0.25          # SIFT works on quarter-resolution images
STRIDE = 32           # point-cloud sampling stride (matches project convention)
SKY_B = 25.0 / 255.0  # sky heuristic on 0..1 RGB
SKY_MIN = 110.0 / 255.0

BUDGET = {
    'photo': 6, 'render': 12, 'closeup': 6, 'match': 5, 'mask': 4,
    'transform': 8, 'solve': 6, 'icp': 4, 'fpfh': 2, 'total': 30,
}


# --------------------------------------------------------------------------- io
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
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def emit(obj):
    print(json.dumps(san(obj), indent=1))


def sdir(case):
    return SESS / case


class SessionLock:
    """Cross-process lock so parallel agent tool calls cannot lose state updates."""

    def __init__(self, case, timeout=180.0):
        self.path = sdir(case) / '.lock'
        self.timeout = timeout
        self.held = False

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.held = True
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > 300:
                        os.unlink(self.path)
                except OSError:
                    pass
                if time.time() - t0 > self.timeout:
                    raise TimeoutError('session busy: %s' % self.path)
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self.held:
            try:
                os.unlink(self.path)
            except OSError:
                pass
            self.held = False
        return False


def load_sess(case):
    p = sdir(case) / 'session.json'
    if not p.exists():
        raise SystemExit('unknown session: %s' % case)
    return json.loads(p.read_text())


def save_sess(sess):
    p = sdir(sess['case']) / 'session.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(san(sess), indent=1))
    tmp.replace(p)


def trace(sess, op, params, result=None, images=None, before=None, after=None, charged=True):
    sess['step'] = int(sess.get('step', 0)) + 1
    rec = {'step': sess['step'], 'op': op, 'params': san(params),
           'charged': bool(charged), 'images': san(images or []),
           'matrix_before': san(before), 'matrix_after': san(after),
           'result': san(result), 'time': time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(sdir(sess['case']) / 'trace.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    save_sess(sess)
    return rec


def charge(sess, kind, n=1):
    b, u = sess['budget'], sess['used']
    if u.get('total', 0) + n > b['total']:
        return False
    if u.get(kind, 0) + n > b[kind]:
        return False
    u['total'] = u.get('total', 0) + n
    u[kind] = u.get(kind, 0) + n
    return True


def budget_error(sess, kind):
    return {'error': 'budget exhausted for %s' % kind,
            'used': sess['used'], 'budget': sess['budget']}


# ------------------------------------------------------------------------- state
def pose(sess, sid):
    p = sess['poses'][sid]
    return np.asarray(p['R'], float), np.asarray(p['t'], float), float(p['s'])


def matrix(sess, sid):
    R, t, s = pose(sess, sid)
    M = np.eye(4)
    M[:3, :3] = R * s
    M[:3, 3] = t
    return M


def set_pose(sess, sid, R=None, t=None, s=None):
    R0, t0, s0 = pose(sess, sid)
    sess['poses'][sid] = {
        'R': (R0 if R is None else np.asarray(R, float)).tolist(),
        't': (t0 if t is None else np.asarray(t, float)).tolist(),
        's': float(s0 if s is None else s),
    }


def push_history(sess):
    sess.setdefault('history', []).append(
        {sid: {'R': p['R'], 't': p['t'], 's': p['s']}
         for sid, p in sess['poses'].items()})


def apply_increment(sess, sid, R=None, t=None, s=None):
    """Left-compose an incremental similarity onto the moving cloud."""
    Ri = np.eye(3) if R is None else np.asarray(R, float)
    ti = np.zeros(3) if t is None else np.asarray(t, float)
    si = 1.0 if s is None else float(s)
    Rold, told, sold = pose(sess, sid)
    push_history(sess)
    set_pose(sess, sid, R=Ri @ Rold, t=si * (Ri @ told) + ti, s=si * sold)


def apply_absolute(sess, sid, R, t, s=1.0):
    """Replace the cumulative transform.  Logged increment is Q = M_new @ M_old^-1."""
    Mold = matrix(sess, sid)
    Mnew = np.eye(4)
    Mnew[:3, :3] = np.asarray(R, float) * float(s)
    Mnew[:3, 3] = np.asarray(t, float)
    Q = Mnew @ np.linalg.inv(Mold)
    push_history(sess)
    set_pose(sess, sid, R=np.asarray(R, float), t=np.asarray(t, float), s=float(s))
    return Q


# -------------------------------------------------------------------------- data
_CLOUD = {}


def load_cloud(scene, sid):
    key = (scene, sid)
    if key in _CLOUD:
        return _CLOUD[key]
    d = DATA / scene
    meta = json.loads((d / 'meta.json').read_text())
    im = next(x for x in meta['images'] if x['sid'] == sid)
    depth = np.load(d / im['depth_file'])
    K = np.asarray(im['K'], float)
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
    img = cv2.cvtColor(cv2.imread(str(d / im['rgb_file'])), cv2.COLOR_BGR2RGB)
    col = img[ys, xs].astype(float) / 255.0
    c = {'scene': scene, 'sid': sid, 'points': P, 'pixels': np.stack([xs, ys], 1),
         'colors': col, 'K': K, 'W': W, 'H': H, 'depth': depth, 'rgb': img}
    _CLOUD[key] = c
    return c


def transformed(cloud, sess):
    R, t, s = pose(sess, cloud['sid'])
    return (cloud['points'] * s) @ R.T + t


def scene_dir(scene):
    return DATA / scene


def mask_keep(sess, cloud):
    keep = np.ones(len(cloud['points']), bool)
    md = sess['mask'].get('max_dist')
    if md is not None:
        keep &= np.linalg.norm(cloud['points'], axis=1) < float(md)
    if sess['mask'].get('drop_sky'):
        c = cloud['colors']
        keep &= ~((c[:, 2] > c[:, 0] + SKY_B) & (c[:, 2] > SKY_MIN) & (c[:, 1] > c[:, 0]))
    return keep


def sky_pixels(rgb, uv):
    H, W = rgb.shape[:2]
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    c = rgb[v, u].astype(float) / 255.0
    return (c[:, 2] > c[:, 0] + SKY_B) & (c[:, 2] > SKY_MIN) & (c[:, 1] > c[:, 0])


# ------------------------------------------------------------------- observations
def overlap_stats(sess, sid, radius=0.35):
    ref_sid = sess['ref']
    cr, cm = load_cloud(sess['scene'], ref_sid), load_cloud(sess['scene'], sid)
    pr = transformed(cr, sess)[mask_keep(sess, cr)]
    pm = transformed(cm, sess)[mask_keep(sess, cm)]
    if len(pr) < 20 or len(pm) < 20:
        return {'moving_near_ref_frac': 0.0, 'ref_near_moving_frac': 0.0,
                'median_nn_m': None, 'n_ref': int(len(pr)), 'n_moving': int(len(pm))}
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pr).query(pm, k=1)
    d2, _ = cKDTree(pm).query(pr, k=1)
    return {'moving_near_ref_frac': float((d < radius).mean()),
            'ref_near_moving_frac': float((d2 < radius).mean()),
            'median_nn_m': float(np.median(d)),
            'n_ref': int(len(pr)), 'n_moving': int(len(pm))}


def state_of(sess):
    return {sid: {'R': p['R'], 't': p['t'], 's': p['s']}
            for sid, p in sess['poses'].items()}


def save_fig(fig, path):
    fig.savefig(path, bbox_inches='tight', dpi=110)
    plt.close(fig)
    return str(path)


def render_state(case, sess, camera='iso', textured=False, size=(7.2, 7.2)):
    ref_sid, tar_sid = sess['ref'], sess['target']
    cr, cm = load_cloud(sess['scene'], ref_sid), load_cloud(sess['scene'], tar_sid)
    kr, km = mask_keep(sess, cr), mask_keep(sess, cm)
    pr = transformed(cr, sess)[kr]
    pm = transformed(cm, sess)[km]
    cr_col = cr['colors'][kr] if textured else np.tile([0.55, 0.55, 0.55], (len(pr), 1))
    cm_col = cm['colors'][km] if textured else np.tile([0.85, 0.15, 0.15], (len(pm), 1))
    rng = np.random.default_rng(0)

    def thin(P, C, n=4000):
        if len(P) <= n:
            return P, C
        idx = rng.choice(len(P), n, replace=False)
        return P[idx], C[idx]

    pr, cr_col = thin(pr, cr_col)
    pm, cm_col = thin(pm, cm_col)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pr[:, 0], pr[:, 1], pr[:, 2], s=1.6, c=cr_col, label='reference (fixed)')
    ax.scatter(pm[:, 0], pm[:, 1], pm[:, 2], s=1.6, c=cm_col, label='moving (current)')
    allp = np.vstack([pr, pm])
    c = np.median(allp, 0)
    r = np.percentile(np.linalg.norm(allp - c, axis=1), 92)
    r = max(r, 0.5)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    cams = {'iso': (28, -55), 'top': (88, -90), 'side': (4, 0), 'front': (12, -90),
            'back': (12, 90), 'left': (12, 0)}
    ax.view_init(*cams.get(camera, cams['iso']))
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title('camera=%s  grey/photo=reference  red/photo=moving current' % camera)
    ax.legend(loc='upper right', fontsize=8)
    p = sdir(case) / ('render_%03d_%s.png' % (sess['step'] + 1, camera))
    fig.tight_layout()
    return save_fig(fig, p)


def draw_grid(img, scale):
    out = img.copy()
    H, W = out.shape[:2]
    step = 500
    for u in range(0, 6049, step):
        x = int(round(u * scale))
        if x >= W:
            break
        cv2.line(out, (x, 0), (x, H - 1), (0, 255, 255), 1)
        cv2.putText(out, str(u), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 255), 1, cv2.LINE_AA)
    for v in range(0, 4033, step):
        y = int(round(v * scale))
        if y >= H:
            break
        cv2.line(out, (0, y), (W - 1, y), (0, 255, 255), 1)
        cv2.putText(out, str(v), (2, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return out


def photo_pair(case, sess):
    ref_sid, tar_sid = sess['ref'], sess['target']
    cr = load_cloud(sess['scene'], ref_sid)
    cm = load_cloud(sess['scene'], tar_sid)
    h = 520
    def panel(rgb):
        sc = h / rgb.shape[0]
        return cv2.resize(rgb, (int(rgb.shape[1] * sc), h), interpolation=cv2.INTER_AREA), sc
    a, sa = panel(cr['rgb'])
    b, sb = panel(cm['rgb'])
    a, b = draw_grid(a, sa), draw_grid(b, sb)
    cv2.putText(a, 'REF %s' % ref_sid, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(b, 'MOVING %s' % tar_sid, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    gap = np.full((h, 12, 3), 255, np.uint8)
    out = np.hstack([a, gap, b])
    p = sdir(case) / ('photos_%03d.png' % (sess['step'] + 1))
    cv2.imwrite(str(p), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return str(p), {'ref_panel': 'left', 'moving_panel': 'right', 'display_height': h,
                    'full_resolution': [int(cr['W']), int(cr['H'])],
                    'grid_step_full_res_px': 500}


def photo_one(case, sess, which):
    sid = sess['ref'] if which == 'ref' else sess['target']
    c = load_cloud(sess['scene'], sid)
    h = 620
    sc = h / c['rgb'].shape[0]
    img = cv2.resize(c['rgb'], (int(c['rgb'].shape[1] * sc), h), interpolation=cv2.INTER_AREA)
    img = draw_grid(img, sc)
    cv2.putText(img, '%s %s' % (which.upper(), sid), (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    p = sdir(case) / ('photo_%s_%03d.png' % (which, sess['step'] + 1))
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(p)


def closeup(case, sess, which, box):
    sid = sess['ref'] if which == 'ref' else sess['target']
    c = load_cloud(sess['scene'], sid)
    u0, v0, u1, v1 = [int(round(x)) for x in box]
    u0, u1 = sorted((max(0, min(c['W'] - 1, u0)), max(0, min(c['W'] - 1, u1))))
    v0, v1 = sorted((max(0, min(c['H'] - 1, v0)), max(0, min(c['H'] - 1, v1))))
    crop = c['rgb'][v0:v1 + 1, u0:u1 + 1]
    if crop.size == 0:
        return {'error': 'empty crop'}, None
    h = 420
    sc = h / max(crop.shape[0], 1)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), h), interpolation=cv2.INTER_AREA)
    p = sdir(case) / ('closeup_%s_%03d.png' % (which, sess['step'] + 1))
    cv2.imwrite(str(p), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    z = c['depth'][v0:v1 + 1, u0:u1 + 1]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    fx, fy, cx, cy = c['K'][0, 0], c['K'][1, 1], c['K'][0, 2], c['K'][1, 2]
    ys, xs = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    P = np.stack([(xs[ok] - cx) / fx * z[ok], (ys[ok] - cy) / fy * z[ok], z[ok]], 1)
    if len(P):
        R, t, s = pose(sess, sid)
        W = s * (P @ R.T) + t
        st = {'n_points': int(len(P)), 'centroid_world_m': W.mean(0),
              'extent_world_m': W.max(0) - W.min(0),
              'median_depth_m': float(np.median(P[:, 2]))}
    else:
        st = {'n_points': 0}
    st.update({'sid': sid, 'box_full_res_px': [u0, v0, u1, v1]})
    return st, str(p)


# ---------------------------------------------------------------------- matching
_DET = {}


def detect(scene, sid, matcher):
    key = (scene, sid, matcher)
    if key in _DET:
        return _DET[key]
    c = load_cloud(scene, sid)
    g = cv2.cvtColor(c['rgb'], cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    if matcher == 'sift':
        det, norm = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02), cv2.NORM_L2
    elif matcher == 'orb':
        det, norm = cv2.ORB_create(nfeatures=4000, fastThreshold=12), cv2.NORM_HAMMING
    elif matcher == 'akaze':
        det, norm = cv2.AKAZE_create(threshold=0.0008), cv2.NORM_HAMMING
    else:
        raise ValueError('unknown matcher')
    kp, des = det.detectAndCompute(g, None)
    _DET[key] = (kp, des, norm)
    return _DET[key]


def backproject(c, uv_full):
    H, W = c['depth'].shape
    u = np.clip(np.round(uv_full[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv_full[:, 1]).astype(int), 0, H - 1)
    z = c['depth'][v, u].astype(float)
    ok = np.isfinite(z) & (z > 0.1) & (z < 200)
    fx, fy, cx, cy = c['K'][0, 0], c['K'][1, 1], c['K'][0, 2], c['K'][1, 2]
    P = np.full((len(uv_full), 3), np.nan)
    P[ok, 0] = (uv_full[ok, 0] - cx) / fx * z[ok]
    P[ok, 1] = (uv_full[ok, 1] - cy) / fy * z[ok]
    P[ok, 2] = z[ok]
    return P, ok


def match_pairs(sess, matcher='sift', ref_box=None, mov_box=None):
    scene, ref_sid, tar_sid = sess['scene'], sess['ref'], sess['target']
    cr, cm = load_cloud(scene, ref_sid), load_cloud(scene, tar_sid)
    k1, d1, norm = detect(scene, ref_sid, matcher)
    k2, d2, _ = detect(scene, tar_sid, matcher)
    if d1 is None or d2 is None:
        return None, {'error': 'no descriptors'}
    bf = cv2.BFMatcher(norm)
    m = bf.knnMatch(d1, d2, k=2)
    good = [a for a, b in m if len((a, b)) == 2 and a.distance < 0.8 * b.distance]
    m2 = bf.knnMatch(d2, d1, k=2)
    back = {a.queryIdx: a.trainIdx for a, b in m2
            if len((a, b)) == 2 and a.distance < 0.8 * b.distance}
    good = [a for a in good if back.get(a.trainIdx) == a.queryIdx]
    uv_r = np.array([k1[a.queryIdx].pt for a in good], float) / SCALE
    uv_m = np.array([k2[a.trainIdx].pt for a in good], float) / SCALE
    if ref_box is not None:
        b = np.asarray(ref_box, float)
        k = ((uv_r[:, 0] >= b[0]) & (uv_r[:, 0] <= b[2]) &
             (uv_r[:, 1] >= b[1]) & (uv_r[:, 1] <= b[3]))
        uv_r, uv_m = uv_r[k], uv_m[k]
    if mov_box is not None:
        b = np.asarray(mov_box, float)
        k = ((uv_m[:, 0] >= b[0]) & (uv_m[:, 0] <= b[2]) &
             (uv_m[:, 1] >= b[1]) & (uv_m[:, 1] <= b[3]))
        uv_r, uv_m = uv_r[k], uv_m[k]
    if len(uv_r) == 0:
        return None, {'error': 'no matches inside the requested boxes'}
    Pr, okr = backproject(cr, uv_r)
    Pm, okm = backproject(cm, uv_m)
    ok = okr & okm
    Pr, Pm, uv_r, uv_m = Pr[ok], Pm[ok], uv_r[ok], uv_m[ok]
    if len(Pr) < 4:
        return None, {'error': 'too few valid 3D correspondences',
                      'n_matches': int(len(uv_r))}
    R, t, s = pose(sess, tar_sid)
    q = s * (Pm @ R.T) + t
    res = np.linalg.norm(Pr - q, axis=1)
    ti = Pr - s * (Pm @ R.T)
    tm = np.median(ti, 0)
    spread = np.median(np.linalg.norm(ti - tm, axis=1))
    data = {'uv_ref': uv_r, 'uv_mov': uv_m, 'P_ref': Pr, 'P_mov': Pm,
            'matcher': matcher}
    f = sdir(sess['case']) / ('matches_%03d.npz' % (sess['step'] + 1))
    np.savez_compressed(f, **data)
    sess['matches'] = {'file': f.name, 'n': int(len(Pr)), 'matcher': matcher}
    return {'file': f.name, 'n': int(len(Pr)), 'n_keypoints': [len(k1), len(k2)],
            'median_residual_under_current_transform_m': float(np.median(res)),
            'implied_translation_spread_m': float(spread)}, None


def match_overlay(sess, matcher, max_draw=70):
    """Draw the stored matches on the side-by-side photographs."""
    scene, ref_sid, tar_sid = sess['scene'], sess['ref'], sess['target']
    cr, cm = load_cloud(scene, ref_sid), load_cloud(scene, tar_sid)
    p = sdir(sess['case']) / sess['matches']['file']
    d = np.load(p)
    uv_r, uv_m = d['uv_ref'], d['uv_mov']
    h = 480
    sc = h / cr['rgb'].shape[0]
    a = cv2.resize(cr['rgb'], (int(cr['rgb'].shape[1] * sc), h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(cm['rgb'], (int(cm['rgb'].shape[1] * sc), h), interpolation=cv2.INTER_AREA)
    gap = np.full((h, 8, 3), 255, np.uint8)
    out = np.hstack([a, gap, b])
    rng = np.random.default_rng(1)
    idx = np.arange(len(uv_r))
    if len(idx) > max_draw:
        idx = rng.choice(idx, max_draw, replace=False)
    off = a.shape[1] + gap.shape[1]
    for i in idx:
        u1, v1 = int(uv_r[i, 0] * sc), int(uv_r[i, 1] * sc)
        u2, v2 = int(uv_m[i, 0] * sc) + off, int(uv_m[i, 1] * sc)
        col = (0, 255, 0) if i % 3 == 0 else (255, 160, 0)
        cv2.circle(out, (u1, v1), 3, col, -1)
        cv2.circle(out, (u2, v2), 3, col, -1)
        cv2.line(out, (u1, v1), (u2, v2), col, 1)
    out = draw_grid(out, sc)
    pp = sdir(sess['case']) / ('matches_%03d.png' % (sess['step'] + 1))
    cv2.imwrite(str(pp), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return str(pp)


def correspondence_mask(sess, d):
    uv_r, uv_m = d['uv_ref'], d['uv_mov']
    cr, cm = load_cloud(sess['scene'], sess['ref']), load_cloud(sess['scene'], sess['target'])
    keep = np.ones(len(uv_r), bool)
    md = sess['mask'].get('max_dist')
    if md is not None:
        keep &= np.linalg.norm(d['P_ref'], axis=1) < float(md)
        keep &= np.linalg.norm(d['P_mov'], axis=1) < float(md)
    if sess['mask'].get('drop_sky'):
        keep &= ~(sky_pixels(cr['rgb'], uv_r) | sky_pixels(cm['rgb'], uv_m))
    return keep


# ----------------------------------------------------------------------- solvers
def kabsch(A, B, with_scale=False):
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = float(S.sum() / max(((A - ca) ** 2).sum(), 1e-12)) if with_scale else 1.0
    return s, R, cb - s * R @ ca


def ransac_rigid(Pm, Pr, iters=3000, tol=0.12, seed=0):
    rng = np.random.default_rng(seed)
    n = len(Pm)
    if n < 4:
        return None
    best = (-1, None)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = kabsch(Pm[idx], Pr[idx], False)
        except np.linalg.LinAlgError:
            continue
        r = np.linalg.norm(Pm @ R.T + t - Pr, axis=1)
        k = int((r < tol).sum())
        if k > best[0]:
            best = (k, np.where(r < tol)[0])
    if best[1] is None or len(best[1]) < 4:
        return None
    s, R, t = kabsch(Pm[best[1]], Pr[best[1]], False)
    return {'R': R, 't': t, 's': 1.0, 'inliers': int(len(best[1])), 'n': n}


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def correction_fit(Pr, Pm, R0, iters=8):
    w = np.zeros(3)
    t = np.median(Pr - Pm @ R0.T, axis=0)
    for _ in range(iters):
        q = Pm @ R0.T
        A = np.zeros((3 * len(q), 6))
        b = (Pr - q).ravel()
        for i, (qx, qy, qz) in enumerate(q):
            A[3 * i:3 * i + 3, 0:3] = [[0, qz, -qy], [-qz, 0, qx], [qy, -qx, 0]]
            A[3 * i:3 * i + 3, 3:6] = np.eye(3)
        res0 = np.linalg.norm((Pr - q) - (np.cross(w, q) + t), axis=1)
        sc = max(float(np.median(res0)), 1e-3)
        wts = 1.0 / (1.0 + (res0 / (2.0 * sc)) ** 2)
        sol, *_ = np.linalg.lstsq(A * np.repeat(wts, 3)[:, None],
                                  b * np.repeat(wts, 3), rcond=None)
        w, t = sol[:3], sol[3:]
    return rodrigues(w) @ R0, t, float(np.degrees(np.linalg.norm(w)))


def solve(kind, sess):
    if sess.get('matches') is None:
        return {'error': 'no correspondences stored; run match first'}
    d = np.load(sdir(sess['case']) / sess['matches']['file'])
    keep = correspondence_mask(sess, d)
    Pr, Pm = d['P_ref'][keep], d['P_mov'][keep]
    if len(Pr) < 6:
        return {'error': 'too few correspondences after the active mask', 'n': int(len(Pr))}
    sid = sess['target']
    R0, t0, s0 = pose(sess, sid)
    if kind == 'translation':
        ti = Pr - s0 * (Pm @ R0.T)
        tm = np.median(ti, axis=0)
        res = np.linalg.norm(ti - tm, axis=1)
        kk = res < max(0.15, np.quantile(res, 0.6))
        t = np.median(ti[kk], axis=0) if kk.sum() >= 3 else tm
        Q = apply_absolute(sess, sid, R0, t, s0)
        spread = float(np.median(np.linalg.norm(Pr - (s0 * (Pm @ R0.T) + t), axis=1)))
        return {'solver': 'translation', 'n_used': int(len(Pr)), 'kept': int(kk.sum()),
                'spread_m': spread, 'increment': Q, 'matrix': matrix(sess, sid)}
    if kind == 'rigid3d':
        fit = ransac_rigid(Pm, Pr)
        if fit is None:
            return {'error': 'RANSAC found no rigid consensus'}
        Q = apply_absolute(sess, sid, fit['R'], fit['t'], 1.0)
        return {'solver': 'rigid3d', 'n_used': int(len(Pr)), 'inliers': fit['inliers'],
                'inlier_fraction': fit['inliers'] / max(len(Pr), 1),
                'increment': Q, 'matrix': matrix(sess, sid)}
    if kind == 'rotcorr':
        R1, t1, deg = correction_fit(Pr, Pm, R0)
        before = float(np.median(np.linalg.norm(Pr - (s0 * (Pm @ R0.T) + t0), axis=1)))
        Q = apply_absolute(sess, sid, R1, t1, s0)
        after = float(np.median(np.linalg.norm(Pr - (s0 * (Pm @ R1.T) + t1), axis=1)))
        return {'solver': 'rotcorr', 'n_used': int(len(Pr)), 'correction_deg': deg,
                'spread_before_m': before, 'spread_after_m': after,
                'increment': Q, 'matrix': matrix(sess, sid)}
    if kind == 'reproj':
        return solve_reproj(sess, Pr, Pm, R0, t0, s0)
    return {'error': 'unknown solver'}


def solve_reproj(sess, Pr, Pm, R0, t0, s0):
    from scipy.optimize import least_squares
    c = load_cloud(sess['scene'], sess['ref'])
    d = np.load(sdir(sess['case']) / sess['matches']['file'])
    keep = correspondence_mask(sess, d)
    uv = d['uv_ref'][keep]
    K = c['K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def resid(p):
        R = rodrigues(p[:3]) @ R0
        q = s0 * (Pm @ R.T) + p[3:6]
        z = np.clip(q[:, 2], 1e-3, None)
        return np.concatenate([fx * q[:, 0] / z + cx - uv[:, 0],
                               fy * q[:, 1] / z + cy - uv[:, 1]])

    x0 = np.concatenate([np.zeros(3), t0])
    sol = least_squares(resid, x0, loss='soft_l1', f_scale=8.0, max_nfev=200)
    R1 = rodrigues(sol.x[:3]) @ R0
    t1 = sol.x[3:6]
    Q = apply_absolute(sess, sess['target'], R1, t1, s0)
    rms = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    return {'solver': 'reproj', 'n_used': int(len(Pr)), 'rms_px': rms,
            'rot_correction_deg': float(np.degrees(np.linalg.norm(sol.x[:3]))),
            'increment': Q, 'matrix': matrix(sess, sess['target'])}


# ------------------------------------------------------------------------ actions
def o3d_cloud(sess, sid):
    import open3d as o3d
    c = load_cloud(sess['scene'], sid)
    keep = mask_keep(sess, c)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(transformed(c, sess)[keep])
    pc.colors = o3d.utility.Vector3dVector(c['colors'][keep])
    return pc


def run_icp(sess, radius=0.6, max_iter=80):
    import open3d as o3d
    src = o3d_cloud(sess, sess['target'])
    dst = o3d_cloud(sess, sess['ref'])
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=20))
    r = o3d.pipelines.registration.registration_icp(
        src, dst, radius, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    T = r.transformation
    newM = T @ matrix(sess, sess['target'])
    Sc = float(np.cbrt(abs(np.linalg.det(newM[:3, :3]))))
    Q = apply_absolute(sess, sess['target'], newM[:3, :3] / Sc, newM[:3, 3], Sc)
    return {'fitness': float(r.fitness), 'rmse': float(r.inlier_rmse),
            'increment': Q, 'matrix': matrix(sess, sess['target'])}


def run_fpfh(sess):
    import open3d as o3d
    o3d.utility.random.seed(0)
    src = o3d_cloud(sess, sess['target'])
    dst = o3d_cloud(sess, sess['ref'])
    fs = src.voxel_down_sample(0.80)
    ft = dst.voxel_down_sample(0.80)
    fs.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    ft.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    Fs = o3d.pipelines.registration.compute_fpfh_feature(
        fs, o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=60))
    Ft = o3d.pipelines.registration.compute_fpfh_feature(
        ft, o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=60))
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        fs, ft, Fs, Ft, True, 1.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [],
        o3d.pipelines.registration.RANSACConvergenceCriteria(12000, 0.95))
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    fine = o3d.pipelines.registration.registration_icp(
        src, dst, 0.5, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    newM = fine.transformation @ matrix(sess, sess['target'])
    Sc = float(np.cbrt(abs(np.linalg.det(newM[:3, :3]))))
    Q = apply_absolute(sess, sess['target'], newM[:3, :3] / Sc, newM[:3, 3], Sc)
    return {'fpfh_ransac_fitness': float(coarse.fitness),
            'icp_fitness': float(fine.fitness), 'icp_rmse': float(fine.inlier_rmse),
            'increment': Q, 'matrix': matrix(sess, sess['target'])}


def rotate(sess, axis, deg):
    ax = {'x': 0, 'y': 1, 'z': 2}[axis]
    th = np.deg2rad(float(deg))
    k = np.zeros(3); k[ax] = 1.0
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    sid = sess['target']
    M0 = matrix(sess, sid)
    apply_increment(sess, sid, R=R, t=np.zeros(3), s=1.0)
    Q = matrix(sess, sid) @ np.linalg.inv(M0)
    return {'rot_axis': axis, 'deg': float(deg), 'increment_matrix': Q,
            'matrix_before': M0, 'matrix': matrix(sess, sid)}


def translate(sess, d):
    sid = sess['target']
    M0 = matrix(sess, sid)
    apply_increment(sess, sid, t=np.asarray(d, float))
    Q = matrix(sess, sid) @ np.linalg.inv(M0)
    return {'delta_m': d, 'increment_matrix': Q, 'matrix_before': M0,
            'matrix': matrix(sess, sid)}


def undo(sess):
    h = sess.get('history') or []
    if not h:
        return {'ok': False, 'error': 'nothing to undo'}
    poses = h.pop()
    sess['poses'] = poses
    return {'ok': True, 'matrix': matrix(sess, sess['target']), 'remaining': len(h)}


# --------------------------------------------------------------------------- CLI
def main(argv):
    ap = argparse.ArgumentParser(prog='workbench_cli.py')
    ap.add_argument('case')
    ap.add_argument('command')
    ap.add_argument('rest', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv[1:])
    try:
        with SessionLock(a.case):
            sess = load_sess(a.case)
            cmd, rest = a.command, a.rest
            if sess.get('status', 'running') != 'running' and cmd not in ('state', 'log', 'stats'):
                emit({'error': 'session is finished', 'status': sess['status']})
                return 0
            return dispatch(sess, cmd, rest)
    except Exception as e:
        import traceback
        err = {'error': '%s: %s' % (type(e).__name__, e),
               'traceback_tail': traceback.format_exc().splitlines()[-3:]}
        emit(err)
        return 1


def dispatch(sess, cmd, rest):
    case = sess['case']
    if cmd == 'state':
        emit({'case': case, 'scene': sess['scene'], 'ref': sess['ref'],
              'target': sess['target'], 'status': sess['status'],
              'matrix': matrix(sess, sess['target']),
              'state': state_of(sess), 'mask': sess['mask'],
              'used': sess['used'], 'budget': sess['budget'],
              'matches': sess.get('matches')})
        return 0
    if cmd == 'log':
        emit({'last_images': sess.get('last_images', [])}); return 0
    if cmd == 'stats':
        ov = overlap_stats(sess, sess['target'])
        c = load_cloud(sess['scene'], sess['target'])
        rec = {'matrix': matrix(sess, sess['target']), 'scale': sess['poses'][sess['target']]['s'],
               'mask': sess['mask'], **ov}
        trace(sess, 'stats', {}, rec, charged=False)
        emit(rec); return 0
    if cmd == 'photos':
        if not charge(sess, 'photo'):
            emit(budget_error(sess, 'photo')); return 0
        before = matrix(sess, sess['target'])
        p, info = photo_pair(case, sess)
        sess['last_images'] = [p]; save_sess(sess)
        rec = trace(sess, 'photos', {}, info, images=[p], before=before, after=before)
        emit({**info, 'images': [p], 'step': rec['step'], 'used': sess['used']}); return 0
    if cmd == 'photo':
        which = rest[0] if rest else 'ref'
        if which not in ('ref', 'target'):
            emit({'error': 'photo ref|target'}); return 0
        if not charge(sess, 'photo'):
            emit(budget_error(sess, 'photo')); return 0
        before = matrix(sess, sess['target'])
        p = photo_one(case, sess, which)
        sess['last_images'] = [p]; save_sess(sess)
        rec = trace(sess, 'photo', {'which': which}, {}, images=[p], before=before, after=before)
        emit({'which': which, 'images': [p], 'step': rec['step'], 'used': sess['used']}); return 0
    if cmd == 'render':
        textured = '--textured' in rest
        pos = [x for x in rest if not x.startswith('--')]
        cam = pos[0] if pos else 'iso'
        if not charge(sess, 'render'):
            emit(budget_error(sess, 'render')); return 0
        before = matrix(sess, sess['target'])
        p = render_state(case, sess, cam, textured=textured)
        sess['last_images'] = [p]; save_sess(sess)
        ov = overlap_stats(sess, sess['target'])
        rec = trace(sess, 'render', {'camera': cam}, ov, images=[p],
                    before=before, after=before)
        emit({**ov, 'camera': cam, 'images': [p], 'step': rec['step'],
              'used': sess['used']}); return 0
    if cmd == 'closeup':
        which = rest[0]
        box = [float(x) for x in rest[1:5]]
        if len(box) != 4:
            emit({'error': 'closeup ref|target u0 v0 u1 v1'}); return 0
        if not charge(sess, 'closeup'):
            emit(budget_error(sess, 'closeup')); return 0
        before = matrix(sess, sess['target'])
        st, p = closeup(case, sess, which, box)
        imgs = [p] if p else []
        rec = trace(sess, 'closeup', {'which': which, 'box': box}, st,
                    images=imgs, before=before, after=before)
        emit({**st, 'images': imgs, 'step': rec['step'], 'used': sess['used']}); return 0
    if cmd == 'match':
        matcher, rb, mb = 'sift', None, None
        i = 0
        while i < len(rest):
            if rest[i] == '--matcher':
                matcher = rest[i + 1]; i += 2
            elif rest[i] == '--ref-box':
                rb = [float(x) for x in rest[i + 1:i + 5]]; i += 5
            elif rest[i] == '--mov-box':
                mb = [float(x) for x in rest[i + 1:i + 5]]; i += 5
            else:
                i += 1
        if not charge(sess, 'match'):
            emit(budget_error(sess, 'match')); return 0
        before = matrix(sess, sess['target'])
        info, err = match_pairs(sess, matcher, rb, mb)
        if err:
            rec = trace(sess, 'match', {'matcher': matcher}, err, before=before, after=before)
            emit({**err, 'step': rec['step'], 'used': sess['used']}); return 0
        p = match_overlay(sess, matcher)
        info.update({'matcher': matcher, 'images': [p], 'step': sess['step'] + 1,
                     'used': sess['used']})
        trace(sess, 'match', {'matcher': matcher, 'ref_box': rb, 'mov_box': mb},
              info, images=[p], before=before, after=before)
        emit(info); return 0
    if cmd == 'mask':
        md = sess['mask'].get('max_dist')
        ds = sess['mask'].get('drop_sky', False)
        i = 0
        while i < len(rest):
            if rest[i] == '--max-dist':
                md = float(rest[i + 1]); i += 2
            elif rest[i] == '--drop-sky':
                ds = True; i += 1
            elif rest[i] == '--keep-sky':
                ds = False; i += 1
            elif rest[i] == '--clear':
                md, ds = None, False; i += 1
            else:
                i += 1
        if not charge(sess, 'mask'):
            emit(budget_error(sess, 'mask')); return 0
        before = matrix(sess, sess['target'])
        sess['mask'] = {'max_dist': md, 'drop_sky': bool(ds)}
        ov = overlap_stats(sess, sess['target'])
        rec = trace(sess, 'mask', {'max_dist': md, 'drop_sky': bool(ds)}, ov,
                    before=before, after=before)
        emit({'mask': sess['mask'], **ov, 'step': rec['step'], 'used': sess['used']}); return 0
    if cmd in ('rotate', 'translate'):
        if not charge(sess, 'transform'):
            emit(budget_error(sess, 'transform')); return 0
        before = matrix(sess, sess['target'])
        if cmd == 'rotate':
            out = rotate(sess, rest[0], float(rest[1]))
        else:
            out = translate(sess, [float(x) for x in rest[:3]])
        ov = overlap_stats(sess, sess['target'])
        rec = trace(sess, cmd, {'rest': rest}, {**out, **ov},
                    before=before, after=matrix(sess, sess['target']))
        emit({**out, **ov, 'step': rec['step'], 'used': sess['used']}); return 0
    if cmd == 'solve':
        if not charge(sess, 'solve'):
            emit(budget_error(sess, 'solve')); return 0
        before = matrix(sess, sess['target'])
        out = solve(rest[0] if rest else 'translation', sess)
        ov = overlap_stats(sess, sess['target'])
        rec = trace(sess, 'solve', {'kind': rest[0] if rest else ''}, {**out, **ov},
                    before=before, after=matrix(sess, sess['target']))
        emit({**out, **ov, 'step': rec['step'], 'used': sess['used']}); return 0
    if cmd in ('icp', 'fpfh'):
        kind = cmd
        if not charge(sess, kind):
            emit(budget_error(sess, kind)); return 0
        before = matrix(sess, sess['target'])
        if kind == 'icp':
            radius = float(rest[0]) if rest else 0.6
            out = run_icp(sess, radius)
        else:
            out = run_fpfh(sess)
        ov = overlap_stats(sess, sess['target'])
        rec = trace(sess, kind, {}, {**out, **ov}, before=before,
                    after=matrix(sess, sess['target']))
        emit({**out, **ov, 'step': rec['step'], 'used': sess['used']}); return 0
    if cmd == 'undo':
        before = matrix(sess, sess['target'])
        out = undo(sess)
        ov = overlap_stats(sess, sess['target'])
        rec = trace(sess, 'undo', {}, {**out, **ov}, before=before,
                    after=matrix(sess, sess['target']), charged=False)
        emit({**out, **ov, 'step': rec['step']}); return 0
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
        if status not in ('done', 'abstain'):
            emit({'error': 'status must be done or abstain'}); return 0
        sess['status'] = status
        sess['reason'] = reason
        trace(sess, 'finish', {'status': status, 'reason': reason},
              {'matrix': matrix(sess, sess['target'])}, charged=False)
        emit({'status': status, 'reason': reason, 'matrix': matrix(sess, sess['target'])})
        return 0
    emit({'error': 'unknown command', 'commands': [
        'state', 'stats', 'log', 'photos', 'photo', 'render', 'closeup', 'match',
        'mask', 'rotate', 'translate', 'solve', 'icp', 'fpfh', 'undo', 'finish']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
