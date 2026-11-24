"""The robot's own forward kinematics as the metric fact: project the known gripper
geometry into the image and compare to the model's depth there."""
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

# known geometry: the gripper's finger span in the robot model
sim = env.sim
print('robot joints:', sim.model.njnt, 'bodies:', sim.model.nbody)
# the eef site is at a known offset from the last link
eef = sim.data.site_xpos[sim.model.site_name2id('gripper0_grip_site')]
cam_id = sim.model.camera_name2id('robot0_eye_in_hand')
campos = sim.data.cam_xpos[cam_id]
print('eef position (world):', np.round(eef, 4))
print('camera position     :', np.round(campos, 4))
print('distance eef->cam   :', round(float(np.linalg.norm(eef-campos)), 4), 'm')
print()
print('This distance is EXACT (kinematics + sim state). If the model could see the eef,')
print('this would be its metric anchor. The eef is not visible to the wrist camera')
print('(it sits behind/inside it), which is why the wrist view has no built-in ruler.')
env.close()
