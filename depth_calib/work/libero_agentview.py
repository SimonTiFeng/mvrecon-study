"""Agent view sees the arm; the arm's link geometry is an exact metric fact."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np, cv2
from libero.libero.envs import OffScreenRenderEnv
BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
OUT = pathlib.Path('D:/libero_agent'); OUT.mkdir(exist_ok=True)
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview','robot0_eye_in_hand'], camera_depths=True)
env.reset()
recs=[]
for step in range(8):
    act=np.zeros(7); act[0]=0.5; act[3]=0.4 if step%2==0 else -0.2
    obs,_,_,_=env.step(act)
    rgb=obs['agentview_image'].astype(np.uint8)
    dep=obs['agentview_depth'][:, :, 0]
    sid=f'ag_{step:02d}'
    cv2.imwrite(str(OUT/f'{sid}.png'), cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    np.save(OUT/f'{sid}_depth.npy', dep.astype(np.float32))
    cam=env.sim.model.camera_name2id('agentview')
    recs.append({'sid':sid,'cam_pos':env.sim.data.cam_xpos[cam].tolist(),
                 'dmin':float(dep.min()),'dmax':float(dep.max()),'dmed':float(np.median(dep))})
    print(f'  {sid}: d {dep.min():.2f}-{dep.max():.2f} med {np.median(dep):.2f}',flush=True)
fovy=float(env.sim.model.cam_fovy[env.sim.model.camera_name2id('agentview')])
(OUT/'meta.json').write_text(json.dumps({'frames':recs,'fovy':fovy,'H':256,'W':256,
                                          'fx':256/2/np.tan(np.deg2rad(fovy/2))},indent=1))
print('agentview fx',round(256/2/np.tan(np.deg2rad(fovy/2)),1),'fovy',fovy)
env.close()
