"""Synthetic and real-pose test for the T_true / D / cumulative-M convention."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, 'mvrecon/workbench')
from wb import Workbench
sys.path.insert(0, 'mvrecon/eval')
from protocol import setup, score

# Algebraic test: COLMAP world->camera poses imply a camera-j -> camera-ref transform.
# Generate a random world point and prove T maps its camera-j coords to camera-ref coords.
rng = np.random.default_rng(7)
def rand_rot():
    A = rng.normal(size=(3,3)); U,_,Vt=np.linalg.svd(A); R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    return R
Rr,Rj=rand_rot(),rand_rot(); tr,tj=rng.normal(size=3),rng.normal(size=3)
Xw=rng.normal(size=3)
pj=Rj@Xw+tj
pr=Rr@Xw+tr
T=np.eye(4); T[:3,:3]=Rr@Rj.T; T[:3,3]=tr-Rr@Rj.T@tj
assert np.allclose(T[:3,:3]@pj+T[:3,3],pr,atol=1e-10)

# Composition test: start at D, apply A=T@inv(D); accumulated matrix must equal T.
D=np.eye(4); D[:3,:3]=rand_rot(); D[:3,3]=rng.normal(size=3)
A=T@np.linalg.inv(D)
assert np.allclose(A@D,T,atol=1e-10)

# Real ETH3D task test on a small office v2 subset, using the actual workbench.
TASK=json.loads(pathlib.Path('mvrecon/eval_data/task.json').read_text())
t=TASK['office']['v2']; wb=Workbench('mvrecon/agent_data/office',stride=32)
setup(wb,t,t['ref']); sid=t['ids'][1]
initial=wb.clouds[sid].matrix().copy()
Treal = np.asarray(t['T_true'][sid])
initial_score = score(wb, t, sid)
Dreal = np.asarray(t['D_injected'][sid])
expected_initial_rot = float(np.degrees(np.arccos(np.clip(
    (np.trace(Dreal[:3,:3] @ Treal[:3,:3].T)-1)/2, -1, 1))))
assert abs(initial_score['rot_err']-expected_initial_rot) < 1e-8
legacy_A = Treal @ np.linalg.inv(Dreal)
legacy_wrong = float(np.degrees(np.arccos(np.clip(
    (np.trace(Dreal[:3,:3] @ legacy_A[:3,:3].T)-1)/2, -1, 1))))
print('PASS: initial score is D vs T_true = %.2f deg (legacy D vs A would be %.2f deg)'
      % (initial_score['rot_err'], legacy_wrong))
A=np.asarray(t['T_true'][sid])@np.linalg.inv(np.asarray(t['D_injected'][sid]))
# apply A as one incremental operation, correctly left-composed
wb.apply(sid,R=A[:3,:3],t=A[:3,3])
M=wb.clouds[sid].matrix(); Treal=np.asarray(t['T_true'][sid])
sc=score(wb,t,sid)
assert np.allclose(M,Treal,atol=1e-8), (M,Treal)
assert sc['rot_err']<1e-5 and sc['trans_err']<1e-8, sc
print('PASS: COLMAP world->camera formula (random point)')
print('PASS: initial D + incremental A recovers cumulative T_true')
print('PASS: scorer reports rot %.2e, trans %.2e' % (sc['rot_err'],sc['trans_err']))
print('PASS: real office v2 task recovery is exact')
