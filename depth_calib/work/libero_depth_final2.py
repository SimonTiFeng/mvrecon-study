"""Use mujoco.mjv_project to place known objects, then compare zbuf there."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np, mujoco
from libero.libero.envs import OffScreenRenderEnv
BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview'], camera_depths=True)
env.reset(); obs = env.step(np.zeros(7))[0]
z = obs['agentview_depth'][:, :, 0].astype(np.float64)
cam = env.sim.model.camera_name2id('agentview')
cpos = np.asarray(env.sim.data.cam_xpos[cam])

# robosuite's renderer holds the mjv scene; project with the scene's camera
renderer = env.sim._render_context_offscreen if hasattr(env.sim,'_render_context_offscreen') else None
print('renderer attr:', [a for a in dir(env.sim) if 'render' in a.lower()][:6])
env.close()
