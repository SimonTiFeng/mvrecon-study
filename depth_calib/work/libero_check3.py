import sys, os
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview'], camera_depths=True)
env.reset()
cam = env.sim.model.camera_name2id('agentview')
cpos = np.asarray(env.sim.data.cam_xpos[cam])
cm = np.asarray(env.sim.data.cam_xmat[cam]).reshape(3,3)
print('cam pos', np.round(cpos,2))
print('cam xmat rows:'); print(np.round(cm,2))
names=[n for n in env.sim.model.body_names if n]
print(f'{len(names)} bodies')
for n in names[:8]:
    b=env.sim.model.body_name2id(n)
    p=np.asarray(env.sim.data.body_xpos[b])
    print(f'  {n:<28} world {np.round(p,2)}  camT*(p-cpos) {np.round(cm.T@(p-cpos),2)}')
env.close()
