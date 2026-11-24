import json,pathlib,numpy as np,open3d as o3d
P=pathlib.Path('mvrecon_pkg_corrected')
T=json.load(open(P/'evaluation_only/task.json'))
# agent/eval split
assert not (P/'agent_input/office/task.json').exists()
assert not (P/'agent_input/office/gt_transforms.json').exists()
# exact geometric roundtrip for office v2 target
sc,tag='office','v2'; task=T[sc][tag]; sid=task['ids'][1]
raw=o3d.io.read_point_cloud(str(P/'agent_input'/sc/'clouds_own'/f'{sid}.ply'))
start=o3d.io.read_point_cloud(str(P/'evaluation_only'/sc/tag/'start_clouds'/f'{sid}.ply'))
align=o3d.io.read_point_cloud(str(P/'evaluation_only'/sc/tag/'gt_aligned_clouds'/f'{sid}.ply'))
r=np.asarray(raw.points); a=np.asarray(start.points); g=np.asarray(align.points)
D=np.asarray(task['D_injected'][sid]); TT=np.asarray(task['T_true'][sid])
a0=r@D[:3,:3].T+D[:3,3]
g0=r@TT[:3,:3].T+TT[:3,3]
print('agent has GT files?',any('T_true' in p.name or 'task.json' in p.name for p in (P/'agent_input').rglob('*')))
print('views',len(list((P/'agent_input'/sc/'rgb').glob('*.png'))),'fullres img size',o3d.io.read_point_cloud if False else '')
print('raw points',len(r),'start maxerr',float(np.abs(a-a0).max()),'gt maxerr',float(np.abs(g-g0).max()))
print('legacy tar exists',(P/'audit/legacy_pre_fix/mvrecon_handoff.tar.gz').exists())
# result links against current task
base=json.load(open(P/'runs/baseline_corrected.json')); agent=json.load(open(P/'runs/agent_runs_corrected.json'))
print('baseline records',len(base),'agent trace records',len(agent))
assert np.abs(a-a0).max()<1e-6 and np.abs(g-g0).max()<1e-6
print('PASS')
