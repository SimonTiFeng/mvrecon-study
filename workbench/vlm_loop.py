"""A genuine VLM-in-the-loop harness.

`run_agent.py` is a step counter, not a policy: it never looks at an image and assumes that the
same rectangle in two photographs contains corresponding content. This harness exists so that a
vision-language model can actually make the decisions, one step at a time, from what it can see.

The loop is deliberately externalised to disk so every "step" is a real model turn rather than a
function call: the model renders, looks at the PNG, decides, and issues one action. Nothing here
chooses for it.

Agent-visible state only:
  - the two original photographs
  - the current point-cloud state, rendered from a viewpoint the model picks
  - overlap / nearest-neighbour residual / current transform

The reference geometry is NEVER read while acting. `score` is the only function that touches
eval_data, is used once at the end, and its result is written to a clearly separate `_eval` block.

Actions:
  init    <scene> <tag> <sid>     create a session, render the start state
  view    <s> <camera>            render (camera: cam | top | side | iso), return path + stats
  photos  <s>                     the two original photographs side by side
  rotate  <s> <why> <ax> <deg>    rotate the moving cloud about a world axis (x|y|z)
  move    <s> <why> <dx> <dy> <dz>  translate the moving cloud
  solve   <s> <why> [--scale] <ur vr um vm> ...   solve from MODEL-chosen pixel correspondences
  undo    <s> <why>               revert the last state change
  stats   <s>
  score   <s>                     EVAL ONLY: final errors against the reference transform
  log     <s>
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workbench'))
sys.path.insert(0, str(ROOT / 'eval'))
from wb import Workbench  # noqa: E402

SESS = ROOT / 'out' / 'vlm_sessions'
CAMS = {'cam': (18, -75), 'top': (88, -90), 'side': (4, 0), 'iso': (28, -55)}


# ---------------------------------------------------------------- session state
def _dir(name):
    return SESS / name


def _load(name):
    return json.loads((_dir(name) / 'state.json').read_text())


def _save(name, st, tools_state=None):
    (_dir(name) / 'state.json').write_text(json.dumps({**st, **(tools_state or {})}, indent=1))


def _wb(name, st):
    wb = Workbench(ROOT / 'agent_data' / st['scene'], stride=32)
    wb.set_reference(st['ref'])
    for sid, pose in st['poses'].items():
        c = wb.clouds[sid]
        c.R = np.asarray(pose['R'], float)
        c.t = np.asarray(pose['t'], float)
        c.s = float(pose['s'])
    return wb


def _snapshot(wb, st):
    return {sid: {'R': c.R.tolist(), 't': c.t.tolist(), 's': float(c.s)}
            for sid, c in wb.clouds.items()}


# ---------------------------------------------------------------- observation
def cloud_colors(scene, sid, cloud, stride=32):
    """Sample each cloud point's colour from its own photograph.

    Reproduces the pixel ordering used when the cloud was built (strided grid, valid-depth mask),
    so the colours line up with the points.
    """
    import cv2
    img = cv2.imread(str(ROOT / 'agent_data' / scene / (sid + '.png')))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = cloud.depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = cloud.depth[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    ys, xs = ys[ok], xs[ok]
    if len(ys) != len(cloud.points):
        return None
    return img[ys, xs] / 255.0


def render(name, st, camera='iso', tag='state', textured=True):
    wb = _wb(name, st)
    ref_all = wb.clouds[st['ref']].transformed()
    mov_all = wb.clouds[st['target']].transformed()
    kr = _keep_mask(st['scene'], st['ref'], wb.clouds[st['ref']], st)
    km = _keep_mask(st['scene'], st['target'], wb.clouds[st['target']], st)
    ref, mov = ref_all[kr], mov_all[km]
    cref = cloud_colors(st['scene'], st['ref'], wb.clouds[st['ref']]) if textured else None
    cmov = cloud_colors(st['scene'], st['target'], wb.clouds[st['target']]) if textured else None
    rng = np.random.default_rng(0)

    def thin(P, n=3500, C=None):
        if len(P) <= n:
            return P, C
        idx = rng.choice(len(P), n, replace=False)
        return P[idx], (None if C is None else C[idx])

    ref, cref = thin(ref, 3500, cref)
    mov, cmov = thin(mov, 3500, cmov)
    fig = plt.figure(figsize=(9, 9), dpi=110)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=1.4,
               c=(cref if cref is not None else '#888888'), label='reference (fixed)')
    ax.scatter(mov[:, 0], mov[:, 1], mov[:, 2], s=1.4,
               c=(cmov if cmov is not None else '#d62728'), label='moving (current)')
    allp = np.vstack([ref, mov])
    c = np.median(allp, 0)
    r = np.percentile(np.linalg.norm(allp - c, axis=1), 92)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(*CAMS.get(camera, CAMS['iso']))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('view=%s   grey=reference  red=moving   current transform shown' % camera)
    ax.legend(loc='upper right', fontsize=8)
    p = _dir(name) / ('%s_%s.png' % (tag, camera))
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    return p


def photos(name, st):
    """The two original photographs, full grid, using the sheet renderer."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('vat', ROOT / 'eval' / 'vlm_anchor_tool.py')
    vat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vat)
    return vat.sheet(st['scene'], st['tag'], st['target'])


def _keep_mask(scene, sid, cloud, st, stride=32):
    """Which points of this view count as usable geometry, under the agent's declared mask."""
    m = st.get('mask') or {}
    md = m.get('max_dist')
    drop_sky = bool(m.get('drop_sky'))
    keep = np.ones(len(cloud.points), bool)
    if md is not None:
        keep &= np.linalg.norm(cloud.points, axis=1) < float(md)
    if drop_sky:
        import cv2
        img = cv2.cvtColor(cv2.imread(str(ROOT / 'agent_data' / scene / (sid + '.png'))),
                           cv2.COLOR_BGR2RGB)
        H, W = cloud.depth.shape
        ys, xs = np.mgrid[0:H:stride, 0:W:stride]
        z = cloud.depth[ys, xs]
        ok = np.isfinite(z) & (z > 0) & (z < 200)
        ys, xs = ys[ok], xs[ok]
        if len(ys) == len(cloud.points):
            col = img[ys, xs].astype(float)
            sky = (col[:, 2] > col[:, 0] + 25) & (col[:, 2] > 110) & (col[:, 1] > col[:, 0])
            keep &= ~sky
    return keep


def feedback(wb, st):
    from scipy.spatial import cKDTree
    ref_c = wb.clouds[st['ref']]
    mov_c = wb.clouds[st['target']]
    kr = _keep_mask(st['scene'], st['ref'], ref_c, st)
    km = _keep_mask(st['scene'], st['target'], mov_c, st)
    R = ref_c.transformed()[kr]
    Mv = mov_c.transformed()[km]
    if len(R) < 50 or len(Mv) < 50:
        return {'overlap': 0.0, 'median_nn_m': float('inf'), 'n_ref': int(len(R)),
                'n_mov': int(len(Mv)), 'scale': float(mov_c.s),
                'R': mov_c.R.tolist(), 't': mov_c.t.tolist()}
    d, _ = cKDTree(R).query(Mv, k=1)
    frac = float((d < 0.35).mean())
    nn = float(np.median(d))
    return {'overlap': frac, 'median_nn_m': nn,
            'n_ref': int(len(R)), 'n_mov': int(len(Mv)),
            'mask': st.get('mask'), 'scale': float(mov_c.s),
            'R': mov_c.R.tolist(), 't': mov_c.t.tolist()}


# ---------------------------------------------------------------- actions
def act_init(scene, tag, sid, from_baseline=False):
    name = '%s_%s_%s' % (scene, tag, sid)
    _dir(name).mkdir(parents=True, exist_ok=True)
    task = json.loads((ROOT / 'eval_data' / 'task.json').read_text())[scene][tag]
    ref = task['ref']
    poses = {}
    if from_baseline:
        # start from the classical front end's answer: this is a tool the agent may run, and it
        # makes the experiment "can inspection improve the front end's rotation?" rather than
        # "can inspection also do the front end's job?"
        base = json.loads((ROOT / 'out' / 'baseline_rescored.json').read_text())
        row = next(r for r in base if r['method'] == 'A_icp' and r['scene'] == scene
                   and r['tag'] == tag and r['sid'] == sid)
        M = np.asarray(row['matrix_est'], float)
        sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
        for s in task['ids']:
            poses[s] = {'R': np.eye(3).tolist(), 't': [0.0, 0.0, 0.0], 's': 1.0}
        poses[sid] = {'R': (M[:3, :3] / sc).tolist(), 't': M[:3, 3].tolist(), 's': sc}
    else:
        for s, M in task['D_injected'].items():
            M = np.asarray(M, float)
            sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
            poses[s] = {'R': (M[:3, :3] / sc).tolist(), 't': M[:3, 3].tolist(), 's': sc}
    st = {'scene': scene, 'tag': tag, 'target': sid, 'ref': ref,
          'poses': poses, 'history': [], 'log': [], 'step': 0}
    wb = _wb(name, st)
    fb = feedback(wb, st)
    st['log'].append({'step': 0, 'action': 'init', 'feedback': fb})
    _save(name, st)
    p = render(name, st, 'iso', 's0')
    return name, p, fb


def _apply(name, why, fn, action, params):
    st = _load(name)
    wb = _wb(name, st)
    before = _snapshot(wb, st)
    msg = fn(wb, st)
    st = _load(name)
    st['poses'] = _snapshot(wb, st)
    st['history'].append(before)
    st['step'] += 1
    fb = feedback(wb, st)
    st['log'].append({'step': st['step'], 'why': why, 'action': action,
                      'params': params, 'result': msg, 'feedback': fb})
    _save(name, st)
    p = render(name, st, 'iso', 's%d' % st['step'])
    return p, fb, msg


def cmd_rotate(name, why, axis, deg):
    ax = {'x': 0, 'y': 1, 'z': 2}[axis]

    def fn(wb, st):
        c = wb.clouds[st['target']]
        th = np.deg2rad(deg)
        k = np.zeros(3)
        k[ax] = 1.0
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        # left-composition: the rotation is applied about the reference-frame origin
        c.t = R @ c.t
        c.R = R @ c.R
        return 'rotated %.2f deg about %s' % (deg, axis)

    return _apply(name, why, fn, 'rotate', {'axis': axis, 'deg': deg})


def cmd_move(name, why, dx, dy, dz):
    def fn(wb, st):
        c = wb.clouds[st['target']]
        c.t = c.t + np.array([dx, dy, dz], float)
        return 'moved (%.2f, %.2f, %.2f)' % (dx, dy, dz)

    return _apply(name, why, fn, 'move', {'d': [dx, dy, dz]})


def cmd_solve(name, why, pairs, with_scale=False):
    def fn(wb, st):
        ref_c = wb.clouds[st['ref']]
        mov_c = wb.clouds[st['target']]

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

        R = mov_c.R.copy()
        Pr, Pm = [], []
        for k in range(0, len(pairs), 4):
            ur, vr, um, vm = pairs[k:k + 4]
            pr, pm = bp(ur, vr, ref_c), bp(um, vm, mov_c)
            if pr is None or pm is None:
                continue
            Pr.append(pr)
            Pm.append(pm)
        if not Pr:
            return 'no usable pair (depth invalid at those pixels)'
        Pr, Pm = np.array(Pr), np.array(Pm)
        s = 1.0
        if with_scale and len(Pr) >= 2:
            ca, cb = Pm.mean(0), Pr.mean(0)
            s = float(np.median(np.linalg.norm(Pr - cb, axis=1)
                                / np.maximum(np.linalg.norm(Pm - ca, axis=1), 1e-9)))
        # translation from the correspondence, with the CURRENT rotation held fixed
        t = np.median(Pr - s * (Pm @ R.T), axis=0) if len(Pr) > 1 else Pr[0] - s * R @ Pm[0]
        mov_c.s = s
        mov_c.t = t
        return 'solved from %d pair(s), scale %.3f' % (len(Pr), s)

    return _apply(name, why, fn, 'solve', {'n_pairs': len(pairs) // 4,
                                           'with_scale': with_scale})


def cmd_solve_sift(name, why):
    """Let the agent run a classical matcher as a tool, restricted to the geometry it trusts."""
    import importlib.util
    import cv2
    spec = importlib.util.spec_from_file_location('si', ROOT / 'eval' / 'sift_identity.py')
    si = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(si)

    def fn(wb, st):
        ref_c, mov_c = wb.clouds[st['ref']], wb.clouds[st['target']]
        sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        gi, gj = si.load_rgb(st['scene'], st['ref']), si.load_rgb(st['scene'], st['target'])
        k1, d1 = sift.detectAndCompute(gi, None)
        k2, d2 = sift.detectAndCompute(gj, None)
        if d1 is None or d2 is None:
            return 'no features'
        m = bf.knnMatch(d1, d2, k=2)
        good = [a for a, b in m if a.distance < 0.8 * b.distance]
        m2 = bf.knnMatch(d2, d1, k=2)
        back = {a.queryIdx: a.trainIdx for a, b in m2 if a.distance < 0.8 * b.distance}
        good = [a for a in good if back.get(a.trainIdx) == a.queryIdx]
        if len(good) < 8:
            return 'too few matches'
        uv_r = np.array([k1[a.queryIdx].pt for a in good]) / si.SCALE
        uv_m = np.array([k2[a.trainIdx].pt for a in good]) / si.SCALE
        Pr, okr = si.backproject(uv_r, ref_c.depth, ref_c.K)
        Pm, okm = si.backproject(uv_m, mov_c.depth, mov_c.K)
        both = okr & okm
        md = (st.get('mask') or {}).get('max_dist')
        if md is not None:
            both &= (np.linalg.norm(Pm, axis=1) < md) & (np.linalg.norm(Pr, axis=1) < md)
        Pr, Pm = Pr[both], Pm[both]
        if len(Pr) < 8:
            return 'too few matches inside the mask'
        R = mov_c.R.copy()
        ti = Pr - Pm @ R.T
        t = np.median(ti, axis=0)
        mov_c.t = t
        sp = float(np.median(np.linalg.norm(ti - t, axis=1)))
        return 'solved t from %d masked matches, spread %.3f m' % (len(Pr), sp)

    return _apply(name, why, fn, 'solve_sift', {})


def cmd_solve_rot(name, why):
    """Estimate a small rotation correction from the correspondences already held.

    With the current R, each correspondence implies t_i = p_ref - R p_mov. If R is wrong by a small
    rotation w about the reference origin, those implied translations vary systematically with the
    point positions:  p_ref - q = [w]x q + t,  q = R p_mov. That is linear in (w, t), so the
    correction is a 6-unknown least-squares problem -- no search, no ground truth. Robust IRLS
    because correspondences on sky or textureless ground are wrong.
    """
    import importlib.util
    import cv2
    spec = importlib.util.spec_from_file_location('si', ROOT / 'eval' / 'sift_identity.py')
    si = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(si)

    def fn(wb, st):
        ref_c, mov_c = wb.clouds[st['ref']], wb.clouds[st['target']]
        sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        gi, gj = si.load_rgb(st['scene'], st['ref']), si.load_rgb(st['scene'], st['target'])
        k1, d1 = sift.detectAndCompute(gi, None)
        k2, d2 = sift.detectAndCompute(gj, None)
        m = bf.knnMatch(d1, d2, k=2)
        good = [a for a, b in m if a.distance < 0.8 * b.distance]
        m2 = bf.knnMatch(d2, d1, k=2)
        back = {a.queryIdx: a.trainIdx for a, b in m2 if a.distance < 0.8 * b.distance}
        good = [a for a in good if back.get(a.trainIdx) == a.queryIdx]
        uv_r = np.array([k1[a.queryIdx].pt for a in good]) / si.SCALE
        uv_m = np.array([k2[a.trainIdx].pt for a in good]) / si.SCALE
        Pr, okr = si.backproject(uv_r, ref_c.depth, ref_c.K)
        Pm, okm = si.backproject(uv_m, mov_c.depth, mov_c.K)
        md = (st.get('mask') or {}).get('max_dist')
        keep = okr & okm
        if md is not None:
            keep &= (np.linalg.norm(Pm, axis=1) < md) & (np.linalg.norm(Pr, axis=1) < md)
        Pr, Pm = Pr[keep], Pm[keep]
        if len(Pr) < 12:
            return 'too few masked correspondences'

        R0 = mov_c.R.copy()
        w = np.zeros(3)
        t = np.median(Pr - Pm @ R0.T, axis=0)
        for _ in range(8):
            q = Pm @ R0.T
            # rows:  [ -[q]x | I ] [w; t] = p_ref - q
            A = np.zeros((3 * len(q), 6))
            b = (Pr - q).ravel()
            for i, (qx, qy, qz) in enumerate(q):
                A[3 * i:3 * i + 3, 0:3] = [[0, qz, -qy], [-qz, 0, qx], [qy, -qx, 0]]
                A[3 * i:3 * i + 3, 3:6] = np.eye(3)
            res0 = np.linalg.norm((Pr - q) - (np.cross(w, q) + t), axis=1)
            sc = max(np.median(res0), 1e-3)
            wts = 1.0 / (1.0 + (res0 / (2.0 * sc)) ** 2)
            Aw = A * np.repeat(wts, 3)[:, None]
            bw = b * np.repeat(wts, 3)
            sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            w, t = sol[:3], sol[3:]
        th = np.linalg.norm(w)
        if th > 1e-12:
            k = w / th
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            C = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        else:
            C = np.eye(3)
        mov_c.R = C @ R0
        mov_c.t = t
        return ('correction %.2f deg (x %+.2f y %+.2f z %+.2f) from %d masked matches'
                % (np.degrees(th), np.degrees(w[0]), np.degrees(w[1]), np.degrees(w[2]), len(Pr)))

    return _apply(name, why, fn, 'solve_rot', {})


def cmd_undo(name, why):
    st = _load(name)
    if not st['history']:
        return None, None, 'nothing to undo'
    prev = st['history'].pop()
    st['poses'] = prev
    st['step'] += 1
    wb = _wb(name, st)
    fb = feedback(wb, st)
    st['log'].append({'step': st['step'], 'why': why, 'action': 'undo',
                      'feedback': fb})
    _save(name, st)
    p = render(name, st, 'iso', 's%d' % st['step'])
    return p, fb, 'reverted to previous state'


def cmd_score(name):
    """EVAL ONLY -- the single place that reads reference geometry."""
    st = _load(name)
    task = json.loads((ROOT / 'eval_data' / 'task.json').read_text())[st['scene']][st['tag']]
    wb = _wb(name, st)
    M = wb.clouds[st['target']].matrix()
    T = np.asarray(task['T_true'][st['target']], float)
    sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    Re = M[:3, :3] / max(sc, 1e-12)
    rot = float(np.degrees(np.arccos(np.clip((np.trace(Re @ T[:3, :3].T) - 1) / 2, -1, 1))))
    lat = float(np.linalg.norm((M[:3, 3] - T[:3, 3])[:2]))
    full = float(np.linalg.norm(M[:3, 3] - T[:3, 3]))
    out = {'rot_err_deg': rot, 'lateral_err_m': lat, 'trans_err_m': full,
           'steps': st['step']}
    p = _dir(name) / 'score_eval_only.json'
    p.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- CLI
def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, name = argv[1], argv[2]
    if cmd == 'init':
        n, p, fb = act_init(argv[2], argv[3], argv[4],
                            from_baseline=('--baseline' in argv))
        print('session:', n)
        print('render:', p)
        print('start feedback:', json.dumps(fb))
        return 0
    if cmd == 'view':
        st = _load(name)
        p = render(name, st, argv[3] if len(argv) > 3 else 'iso', 'v%d' % st['step'])
        print(p)
        print(json.dumps(feedback(_wb(name, st), st)))
        return 0
    if cmd == 'photos':
        print(photos(name, _load(name)))
        return 0
    if cmd == 'stats':
        st = _load(name)
        print(json.dumps(feedback(_wb(name, st), st)))
        return 0
    if cmd == 'rotate':
        p, fb, msg = cmd_rotate(name, argv[3], argv[4], float(argv[5]))
        print(msg); print(p); print(json.dumps(fb)); return 0
    if cmd == 'move':
        p, fb, msg = cmd_move(name, argv[3], float(argv[4]), float(argv[5]), float(argv[6]))
        print(msg); print(p); print(json.dumps(fb)); return 0
    if cmd == 'solve':
        why = argv[3]
        rest = argv[4:]
        ws = False
        if rest and rest[0] == '--scale':
            ws, rest = True, rest[1:]
        p, fb, msg = cmd_solve(name, why, [float(v) for v in rest], ws)
        print(msg); print(p); print(json.dumps(fb)); return 0
    if cmd == 'mask':
        st = _load(name)
        md = None if argv[3] in ('none', 'None', '-') else float(argv[3])
        ds = '--drop-sky' in argv
        st['mask'] = {'max_dist': md, 'drop_sky': ds}
        st['step'] += 1
        st['log'].append({'step': st['step'], 'why': argv[4] if len(argv) > 4 else '',
                          'action': 'mask', 'params': st['mask'],
                          'feedback': feedback(_wb(name, st), st)})
        _save(name, st)
        print(json.dumps(feedback(_wb(name, st), st)))
        return 0
    if cmd == 'solve_sift':
        p, fb, msg = cmd_solve_sift(name, argv[3])
        print(msg); print(p); print(json.dumps(fb)); return 0
    if cmd == 'solve_rot':
        p, fb, msg = cmd_solve_rot(name, argv[3])
        print(msg); print(p); print(json.dumps(fb)); return 0
    if cmd == 'undo':
        p, fb, msg = cmd_undo(name, argv[3])
        print(msg)
        if p:
            print(p); print(json.dumps(fb))
        return 0
    if cmd == 'score':
        print(json.dumps(cmd_score(name))); return 0
    if cmd == 'log':
        st = _load(name)
        for e in st['log']:
            print(json.dumps({k: e[k] for k in e if k != 'feedback'}))
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
