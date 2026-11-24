"""Is the agentview depth actually correct? Compare to known sim geometry."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview','robot0_eye_in_hand'], camera_depths=True)
env.reset()
cam = env.sim.model.camera_name2id('agentview')
cpos = env.sim.data.cam_xpos[cam]
print('agentview camera pos', np.round(cpos,3))
# known object positions from the sim
for name in ('moka_pot_1_main','chefmate_8_frypan_1_main','flat_stove_1_main'):
    try:
        bid = env.sim.model.body_name2id(name)
        p = env.sim.data.body_xpos[bid]
        print(f'  {name}: pos {np.round(p,3)}  dist to cam {np.linalg.norm(p-cpos):.3f} m')
    except Exception as e:
        print(f'  {name}: n/a')
obs,_,_,_ = env.step(np.zeros(7))
d = obs['agentview_depth'][:, :, 0]
print()
print(f'rendered depth range {d.min():.3f}-{d.max():.3f}')
print('If objects sit 1-2 m from the camera, the depth should span that range.')
env.close()
