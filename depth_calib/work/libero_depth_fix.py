"""Convert MuJoCo's normalised z-buffer to metric depth, then validate against known geometry."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np, cv2
from libero.libero.envs import OffScreenRenderEnv

BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview','robot0_eye_in_hand'], camera_depths=True)
env.reset()
_,_,_,_ = env.step(np.zeros(7))
obs = env.step(np.zeros(7))[0]

cam = env.sim.model.camera_name2id('agentview')
cpos = np.asarray(env.sim.data.cam_xpos[cam])
mp = np.asarray(env.sim.data.body_xpos[env.sim.model.body_name2id('moka_pot_1_main')])
dist_true = float(np.linalg.norm(mp - cpos))

d_norm = obs['agentview_depth'][:, :, 0]
print('raw z-buffer range', round(float(d_norm.min()),5), round(float(d_norm.max()),5))
print('moka pot true distance from camera:', round(dist_true,3), 'm')
print()
print('MuJoCo z-buffer convention: depth = (far*near)/(far - z*(far-near)),')
print('but the near/far used here are the renderer defaults. Solve for them:')
print('  We know the moka pot is at', round(dist_true,3), 'm.')
# take the z value at the moka pot pixel: find it via the RGB centre of the pot
# simplest: use the range of z and the known depth range
print()
print('Instead of guessing, measure against multiple known objects.')
for name in ('moka_pot_1_main','chefmate_8_frypan_1_main','flat_stove_1_main'):
    bid = env.sim.model.body_name2id(name)
    p = np.asarray(env.sim.data.body_xpos[bid])
    print(f'  {name}: true {np.linalg.norm(p-cpos):.3f} m')
env.close()
