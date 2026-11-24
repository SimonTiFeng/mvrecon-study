"""Decisive check: is the depth buffer the metric distance or a normalised value?"""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np, cv2
from libero.libero.envs import OffScreenRenderEnv
BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview'], camera_depths=True)
env.reset(); obs = env.step(np.zeros(7))[0]
z = obs['agentview_depth'][:, :, 0].astype(np.float64)
cam = env.sim.model.camera_name2id('agentview')
cpos = np.asarray(env.sim.data.cam_xpos[cam])
cm = np.asarray(env.sim.data.cam_xmat[cam]).reshape(3,3)
fovy = float(env.sim.model.cam_fovy[cam]); f = 256/2/np.tan(np.deg2rad(fovy/2))
print('camera', np.round(cpos,3), 'fovy', fovy, 'f', round(f,1))
print(f"{'body':<26}{'dist':>8}{'u':>6}{'v':>6}{'zbuf':>9}")
hits=[]
for name in env.sim.model.body_names:
    if not name or name=='world': continue
    p = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(name)])
    d = float(np.linalg.norm(p-cpos))
    rel = cm.T @ (p-cpos)
    if rel[2] >= -0.01: continue
    zc = -rel[2]
    u = f*rel[0]/zc + 128; v = 128 - f*rel[1]/zc
    iu,iv = int(round(u)), int(round(v))
    if 0<=iu<256 and 0<=iv<256:
        zv = float(z[iv,iu])
        hits.append((name,d,zv))
        if len(hits)<=10: print(f"{name:<26}{d:>8.3f}{iu:>6}{iv:>6}{zv:>9.5f}")
print()
if len(hits)>=2:
    ds=np.array([h[1] for h in hits]); zs=np.array([h[2] for h in hits])
    print(f'{len(hits)} bodies project into view; true dist {ds.min():.2f}-{ds.max():.2f} m')
    print(f'zbuf values at those pixels   {zs.min():.5f}-{zs.max():.5f}')
    print('corr(dist, zbuf) =', round(float(np.corrcoef(ds,zs)[0,1]),3))
env.close()
