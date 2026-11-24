"""Verify corrected handoff matrices and agent/eval isolation."""
from __future__ import annotations
import json, pathlib
import numpy as np
import open3d as o3d

P=pathlib.Path(__file__).resolve().parents[1]/'mvrecon_pkg_corrected'
assert (P/'README.md').exists()
assert (P/'audit/legacy_pre_fix/README.md').exists()
assert (P/'audit/COORDINATE_AUDIT.md').exists()
assert (P/'RESULTS.md').exists()
scenes=sorted(x.name for x in (P/'agent_input').iterdir() if x.is_dir())
print('scenes',len(scenes),scenes)

# Isolation check must be structural: forbidden KEYS in JSON, forbidden FILES on disk.
# (A prose disclaimer that names these concepts is not a leak.)
FORBIDDEN_KEYS={'T_true','D_injected','T_true_eval_only','D_injected_eval_only',
                'matrix_gt','initial_matrix','final_matrix','R','t'}
FORBIDDEN_FILES={'task.json','task_gt.json','gt_transforms.json','colmap_meta.json',
                 'subsets.json','meta.json'}

def walk_keys(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(obj,list):
        for v in obj: yield from walk_keys(v)

leaks=[]
for p in (P/'agent_input').rglob('*'):
    if not p.is_file(): continue
    if p.name in FORBIDDEN_FILES:
        leaks.append(('file',str(p))); continue
    if p.suffix=='.json':
        keys=set(walk_keys(json.loads(p.read_text(encoding='utf-8'))))
        hit=keys & FORBIDDEN_KEYS
        if hit: leaks.append(('key',str(p),sorted(hit)))
assert not leaks, leaks
print('agent/eval split: PASS (no forbidden keys or files in agent_input)')

# Check one raw/start/aligned point cloud against the stored D and T.
scene='office'; tag='v2'
task=json.loads((P/'evaluation_only'/scene/tag/'task_gt.json').read_text())
sid=task['ids'][1]
raw=o3d.io.read_point_cloud(str(P/'agent_input'/scene/'clouds_own'/f'{sid}.ply'))
start=o3d.io.read_point_cloud(str(P/'evaluation_only'/scene/tag/'start_clouds'/f'{sid}.ply'))
aligned=o3d.io.read_point_cloud(str(P/'evaluation_only'/scene/tag/'gt_aligned_clouds'/f'{sid}.ply'))
r=np.asarray(raw.points); a=np.asarray(start.points); g=np.asarray(aligned.points)
D=np.asarray(task['D_injected'][sid]); T=np.asarray(task['T_true'][sid])
expect_a=r@D[:3,:3].T + D[:3,3]
expect_g=r@T[:3,:3].T + T[:3,3]
err_a=float(np.max(np.abs(a-expect_a))); err_g=float(np.max(np.abs(g-expect_g)))
assert err_a<1e-6 and err_g<1e-6
print('pointclouds: raw',len(r),'start maxerr',err_a,'GT-align maxerr',err_g)

# Check all saved rows correspond to the current evaluation task and preserve matrices/traces.
TALL=json.loads((P/'evaluation_only/task.json').read_text())
base=json.loads((P/'runs/baseline_corrected.json').read_text())
agent=json.loads((P/'runs/agent_runs_corrected.json').read_text())
for x in base:
    t=TALL[x['scene']][x['tag']]
    assert x['sid'] in t['ids'][1:]
    assert np.max(np.abs(np.asarray(x['matrix_gt'])-np.asarray(t['T_true'][x['sid']])))<1e-8
    assert 'initial_matrix' in x and 'matrix_est' in x and 'method_result' in x
for x in agent:
    t=TALL[x['scene']][x['tag']]
    assert x['sid'] in t['ids'][1:]
    assert np.max(np.abs(np.asarray(x['T_true_eval_only'])-np.asarray(t['T_true'][x['sid']])))<1e-8
    assert 'initial_matrix' in x and 'final_matrix' in x and 'trace' in x and 'hist' in x
print('baseline traces',len(base),'agent traces',len(agent),'task links: PASS')
print('corrected handoff: PASS')
