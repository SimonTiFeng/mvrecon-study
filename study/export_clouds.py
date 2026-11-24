#!/usr/bin/env python
"""Export each agent's aligned moving cloud and the fixed reference cloud.

Produces, per case:
  <scene>/<case>_agent.ply   moving cloud at the agent's final cumulative transform
  <scene>/<case>_ref.ply     reference cloud (fixed)
  <scene>/<case>_input.ply   moving cloud at the injected start transform (box only)

and one self-contained interactive viewer per scene plus an index page.  The viewer
embeds plotly.min.js, so it opens offline in a browser with no server and no network.
Use --all to export every case (the default is one representative case per scene).
"""
from __future__ import annotations

import os
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'out' / 'agent_study_20261123'
SANDBOX = pathlib.Path(os.environ.get('MVREBENCH_SANDBOX',
                                      pathlib.Path(os.environ.get('TEMP', '/tmp')) / 'reg_wb_study_20261123')) / 'sessions'
AG = ROOT / 'agent_data'
STRIDE = 32
PLOTLY_JS = pathlib.Path(__import__('plotly').__file__).parent / 'package_data' / 'plotly.min.js'


def load(sid, scene):
    meta = json.loads((AG / scene / 'meta.json').read_text())
    im = next(x for x in meta['images'] if x['sid'] == sid)
    depth = np.load(AG / scene / im['depth_file'])
    K = np.asarray(im['K'], float)
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], 1)
    return P, depth, im


def rgb(sid, scene, depth):
    import cv2
    img = cv2.cvtColor(cv2.imread(str(AG / scene / (sid + '.png'))), cv2.COLOR_BGR2RGB)
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0) & (z < 200)
    return img[ys[ok], xs[ok]].astype(float) / 255.0


def matrix(pose):
    M = np.eye(4)
    M[:3, :3] = np.asarray(pose['R'], float) * float(pose['s'])
    M[:3, 3] = np.asarray(pose['t'], float)
    return M


def apply(P, M):
    return P @ M[:3, :3].T + M[:3, 3]


def write_ply(path, P, C):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    pc.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    o3d.io.write_point_cloud(str(path), pc, write_ascii=False, compressed=False)


def thin(P, C, n=12000, seed=0):
    if len(P) <= n:
        return P, C
    idx = np.random.default_rng(seed).choice(len(P), n, replace=False)
    return P[idx], C[idx]


def score(case, sess_dir, T):
    s = json.loads((sess_dir / 'session.json').read_text())
    sid = s['target']
    M = matrix(s['poses'][sid])
    T = np.asarray(T[s['scene']][s['tag']]['T_true'][sid], float)
    sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    R = M[:3, :3] / sc
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R @ T[:3, :3].T) - 1) / 2, -1, 1))))
    d = M[:3, 3] - T[:3, 3]
    return M, rot, float(np.linalg.norm(d[:2])), s['status']


def export(cases):
    T = json.loads((ROOT / 'eval_data' / 'task.json').read_text())
    dst = OUT / 'export' / 'clouds'
    scene_files = {}
    index = []
    for case in cases:
        sess_dir = SANDBOX / case
        s = json.loads((sess_dir / 'session.json').read_text())
        scene, ref, tar = s['scene'], s['ref'], s['target']
        M, rot, lat, status = score(case, sess_dir, T)
        P, d, _ = load(tar, scene)
        C = rgb(tar, scene, d)
        Pm = apply(P, M)
        Pr, dr, _ = load(ref, scene)
        Cr = rgb(ref, scene, dr)
        Mi = np.asarray(s['initial_matrix'], float)
        Pi = apply(P, Mi)
        east = OUT / 'export' / 'clouds' / scene
        east.mkdir(parents=True, exist_ok=True)
        write_ply(east / ('%s_agent.ply' % case), Pm, C)
        write_ply(east / ('%s_ref.ply' % case), Pr, Cr)
        write_ply(east / ('%s_input.ply' % case), Pi, C)
        Ps, Cs = thin(Pm, C)
        Prs, Crs = thin(Pr, Cr, seed=1)
        Pis, _ = thin(Pi, C, seed=2)
        rec = {'case': case, 'scene': scene, 'tag': s['tag'], 'ref': ref, 'target': tar,
               'status': status, 'rot_err': rot, 'lat_err': lat,
               'files': {'agent': 'clouds/%s/%s_agent.ply' % (scene, case),
                         'ref': 'clouds/%s/%s_ref.ply' % (scene, case),
                         'input': 'clouds/%s/%s_input.ply' % (scene, case)}}
        index.append(rec)
        scene_files.setdefault(scene, []).append({
            'case': case, 'tag': s['tag'], 'status': status, 'rot': rot, 'lat': lat,
            'Pm': Ps, 'Cm': Cs, 'Pr': Prs, 'Cr': Crs, 'Pi': Pis})
    (OUT / 'export').mkdir(parents=True, exist_ok=True)
    (OUT / 'export' / 'index.json').write_text(json.dumps(index, indent=1))
    (OUT / 'export' / 'clouds.json').write_text(json.dumps(index, indent=1))
    return scene_files, index


def fig_html(scene, items):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    n = len(items)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=['%s  %s  %s  agent %.1f° %.2f m' %
                                        (scene, it['tag'], it['case'].split('_')[-1],
                                         it['rot'], it['lat']) for it in items],
                        specs=[[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)])
    for k, it in enumerate(items):
        r, c = k // cols + 1, k % cols + 1
        fig.add_trace(go.Scatter3d(x=it['Pr'][:, 0], y=it['Pr'][:, 1], z=it['Pr'][:, 2],
                                   mode='markers', name='reference',
                                   marker=dict(size=1.4, color=it['Cr'], opacity=0.55),
                                   showlegend=(k == 0)), row=r, col=c)
        fig.add_trace(go.Scatter3d(x=it['Pm'][:, 0], y=it['Pm'][:, 1], z=it['Pm'][:, 2],
                                   mode='markers', name='agent-aligned',
                                   marker=dict(size=1.4, color=[0.85, 0.15, 0.15],
                                               opacity=0.55), showlegend=(k == 0)),
                      row=r, col=c)
    fig.update_layout(height=620 * rows, margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(itemsizing='constant'))
    for k in range(1, n + 1):
        axis = fig.layout['scene%d' % k] if k > 1 else fig.layout.scene
        axis.xaxis.title = 'x (m)'
        axis.yaxis.title = 'y (m)'
        axis.zaxis.title = 'z (m)'
        axis.aspectmode = 'data'
    return fig


def write_html(path, fig, title, single=False):
    import plotly.offline as off
    if PLOTLY_JS.exists():
        fig.write_html(str(path), include_plotlyjs='inline',
                       config={'displaylogo': False}, auto_open=False)
    else:
        off.plot(fig, filename=str(path), auto_open=False, include_plotlyjs=True,
                 config={'displaylogo': False})
    return path


def index_page(scene_pages, index):
    rows = []
    for r in sorted(index, key=lambda x: (x['scene'], x['tag'])):
        ply = r['files']['agent']
        rows.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
                    '<td>%.1f°</td><td>%.2f m</td><td>%s</td>'
                    '<td><a href="%s">moving cloud</a> · <a href="%s">ref</a></td></tr>'
                    % (r['scene'], r['tag'], r['case'].split('_')[-1], r['status'],
                       r['rot_err'], r['lat_err'], r['target'],
                       r['files']['agent'], r['files']['ref']))
    scenes = ''.join('<li><a href="%s">%s</a> (%d cases)</li>' % (p, s, n)
                     for s, p, n in scene_pages)
    html = ('<!doctype html><html><head><meta charset="utf-8"><title>MV-Recon agent results</title>'
            '<style>body{font-family:system-ui,sans-serif;margin:2em;max-width:1100px}'
            'table{border-collapse:collapse;width:100%%}td,th{border:1px solid #ddd;padding:4px 8px;'
            'font-size:13px}th{background:#f3f3f3}a{color:#2563eb}</style></head><body>'
            '<h1>MV-Recon interactive agent — aligned point clouds</h1>'
            '<p>Red = moving cloud at the agent’s final transform; photo-coloured = reference. '
            'Error is measured against the true camera pose after the agent finished. '
            'PLY files contain full-resolution coloured clouds.</p>'
            '<h2>3D viewers</h2><ul>%s</ul>'
            '<h2>All cases</h2><table><tr><th>scene</th><th>tag</th><th>target</th><th>status</th>'
            '<th>rot err</th><th>lat err</th><th>target id</th><th>files</th></tr>%s</table>'
            '</body></html>' % (scenes, ''.join(rows)))
    p = OUT / 'export' / 'index.html'
    p.write_text(html, encoding='utf-8')
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    order = {'v2': 0, 'v3': 1, 'v4': 2}
    if a.all:
        sess = [p.name for p in SANDBOX.iterdir() if (p / 'session.json').exists()]
        scene_of = {}
        for c in sess:
            s = json.loads((SANDBOX / c / 'session.json').read_text())
            scene_of.setdefault(s['scene'], []).append((order.get(s['tag'], 9), c))
        cases = []
        for scene, lst in sorted(scene_of.items()):
            cases.extend([c for _, c in sorted(lst)])
    else:
        res = json.loads((OUT / 'eval' / 'agent_results.json').read_text())
        by_scene = {}
        for r in res:
            by_scene.setdefault(r['scene'], []).append(r)
        cases = [sorted(rs, key=lambda r: order[r['tag']])[0]['case']
                 for _, rs in sorted(by_scene.items())]
    scene_files, index = export(cases)
    pages = []
    for scene, items in sorted(scene_files.items()):
        fig = fig_html(scene, items)
        p = OUT / 'export' / ('%s.html' % scene)
        write_html(p, fig, scene)
        sub = [r for r in index if r['scene'] == scene]
        (OUT / 'export' / ('%s_cases.json' % scene)).write_text(json.dumps(sub, indent=1))
        pages.append((scene, '%s.html' % scene, len(items)))
    idx = index_page(pages, index)
    print('exported', len(cases), 'cases to', OUT / 'export')
    print('index:', idx)
    print('ply bytes:', sum(f.stat().st_size for f in (OUT / 'export' / 'clouds').rglob('*.ply')))


if __name__ == '__main__':
    main()
