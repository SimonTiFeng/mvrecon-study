"""Use mujoco's own Renderer (validated earlier) instead of robosuite's z-buffer."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np, mujoco
from libero.libero.envs import OffScreenRenderEnv
BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview'], camera_depths=False)
env.reset(); env.step(np.zeros(7))
m = env.sim.model._model
d = env.sim.data._data
r = mujoco.Renderer(m, 256, 256)
r.enable_depth_rendering()
r.update_scene(d, camera='agentview')
dep = r.render()
print('mj depth', dep.shape, 'min', round(float(dep.min()),3), 'max', round(float(dep.max()),3))
cam = env.sim.model.camera_name2id('agentview')
cpos = np.asarray(env.sim.data.cam_xpos[cam])
cm = np.asarray(env.sim.data.cam_xmat[cam]).reshape(3,3)
f = 256/2/np.tan(np.deg2rad(float(env.sim.model.cam_fovy[cam])/2))
print(f'{"body":<26}{"true":>7}{"mj_depth":>10}')
for n in ('moka_pot_1_main','chefmate_8_frypan_1_main','flat_stove_1_main','gripper0_eef'):
    p = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id(n)])
    rel = cm.T @ (p - cpos)
    if rel[2] >= -0.05: continue
    zc = -rel[2]; u = f*rel[0]/zc+128; v = 128 - f*rel[1]/zc
    iu,iv = int(round(u)),int(round(v))
    if 2<=iu<254 and 2<=iv<254:
        print(f'{n:<26}{np.linalg.norm(p-cpos):>7.3f}{float(np.median(dep[iv-1:iv+2,iu-1:iu+2])):>10.3f}')
env.close()
