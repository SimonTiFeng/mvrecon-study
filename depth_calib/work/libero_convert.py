"""Fit near/far for the z-buffer conversion and validate on 22 known bodies."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np
from scipy.optimize import least_squares
from libero.libero.envs import OffScreenRenderEnv
BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview'], camera_depths=True)
env.reset(); obs = env.step(np.zeros(7))[0]
z = obs['agentview_depth'][:, :, 0].astype(np.float64)
cam = env.sim.model.camera_name2id('agentview')
cpos = np.asarray(env.sim.data.cam_xpos[cam])
cm = np.asarray(env.sim.data.cam_xmat[cam]).reshape(3,3)
f = 256/2/np.tan(np.deg2rad(float(env.sim.model.cam_fovy[cam])/2))

pairs=[]
for n in env.sim.model.body_names:
    if not n or n=='world': continue
    p = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(n)])
    rel = cm.T @ (p - cpos)
    if rel[2] >= -0.05: continue
    zc = -rel[2]
    u = f*rel[0]/zc + 128; v = 128 - f*rel[1]/zc
    iu,iv = int(round(u)), int(round(v))
    if 2<=iu<254 and 2<=iv<254:
        pairs.append((float(np.linalg.norm(p-cpos)), float(np.median(z[iv-1:iv+2,iu-1:iu+2]))))
pairs = np.array(pairs)
print(f'{len(pairs)} calibration points; true dist {pairs[:,0].min():.2f}-{pairs[:,0].max():.2f}')

def resid(pr):
    near, far = pr
    d_pred = far*near/((far-near)*pairs[:,1]+near)
    return d_pred - pairs[:,0]
r = least_squares(resid, [0.05, 20.0], bounds=([1e-3,0.5],[0.5,200.0]))
near, far = r.x
print(f'fitted near={near:.4f}  far={far:.4f}')
print(f'residuals: max {np.abs(r.fun).max():.4f} m, rms {np.sqrt((r.fun**2).mean()):.4f} m')
p2 = far*near/((far-near)*pairs[:,1]+near)
print(f'\n{"true":>7}{"converted":>11}{"err":>8}')
for a,b in pairs[:10]: print(f'{a:>7.3f}{b:>11.3f}{b-a:>8.3f}')
print('\nIf rms is a few cm, the conversion is correct and LIBERO depths are usable.')
env.close()
