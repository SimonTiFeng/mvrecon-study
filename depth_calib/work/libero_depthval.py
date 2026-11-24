"""Convert z-buffer to metric depth and validate against 3 known object distances."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np, cv2
from libero.libero.envs import OffScreenRenderEnv

BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview'], camera_depths=True)
env.reset(); obs = env.step(np.zeros(7))[0]
cam = env.sim.model.camera_name2id('agentview')
cpos = np.asarray(env.sim.data.cam_xpos[cam])
z = obs['agentview_depth'][:, :, 0].astype(np.float64)

# project each object's centre to find its pixel, then read z there
fovy = float(env.sim.model.cam_fovy[cam]); H=W=256
fx = H/2/np.tan(np.deg2rad(fovy/2))
cam_mat = np.asarray(env.sim.data.cam_xmat[cam]).reshape(3,3)  # world->cam rot
objs = {'moka_pot_1_main':None,'chefmate_8_frypan_1_main':None,'flat_stove_1_main':None}
for name in objs:
    p = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(name)])
    rel = cam_mat @ (p - cpos)
    if rel[2] <= 0: continue
    u = fx*rel[0]/rel[2] + W/2; v = H/2 - fx*rel[1]/rel[2]
    iu,iv = int(round(u)), int(round(v))
    print(f"  {name}: rel {np.round(rel,3)} uv {u:.0f},{v:.0f}")
    if 0<=iu<W and 0<=iv<H:
        zval = float(np.median(z[max(0,iv-2):iv+3, max(0,iu-2):iu+3]))
        objs[name]=(float(np.linalg.norm(p-cpos)), zval, (iu,iv))
print(f"{'object':<28}{'true dist':>10}{'zbuf':>9}{'pixel':>12}")
pairs=[]
for n,v in objs.items():
    if v: print(f"{n:<28}{v[0]:>10.3f}{v[1]:>9.5f}{str(v[2]):>12}"); pairs.append(v)

# MuJoCo: z_buffer = (far*near)/(far - d*(far-near)) inverted:
#   d = far*near / ((far-near)*z + near) ... solve near,far from the pairs
from scipy.optimize import least_squares
def resid(pr):
    near,far = pr
    out=[]
    for d_true,zb,_ in pairs:
        d_pred = far*near/((far-near)*zb+near)
        out.append(d_pred-d_true)
    return np.array(out)
r = least_squares(resid, [0.1, 10.0], bounds=([0.001,1.0],[1.0,100.0]))
near,far = r.x
print(f"\nfitted near={near:.4f} far={far:.4f}  resid={np.round(r.fun,4)}")
print("\n=> if residuals are small, the conversion works and the earlier LIBERO")
print("   depth numbers were reading the raw z-buffer, not metres.")
env.close()
