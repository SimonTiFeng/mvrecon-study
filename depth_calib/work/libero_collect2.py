"""Collect LIBERO frames with real arm motion, plus exact depth AND camera pose per frame."""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np, cv2
from libero.libero.envs import OffScreenRenderEnv

BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
OUT = pathlib.Path('D:/libero_out2'); OUT.mkdir(exist_ok=True)
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview','robot0_eye_in_hand'], camera_depths=True)
env.reset()
cam_id = env.sim.model.camera_name2id('robot0_eye_in_hand')
fovy = float(env.sim.model.cam_fovy[cam_id])

recs = []
# sweep the arm so the wrist camera sees different geometry
for step in range(12):
    act = np.array([0.6, -0.4 if step%3==0 else 0.0, 0.5 if step%4==0 else 0.0,
                    0.0, 0.0, 0.0, 1.0 if step%5==0 else -1.0])
    obs, _, _, _ = env.step(act)
    rgb = obs['robot0_eye_in_hand_image'].astype(np.uint8)
    dep = obs['robot0_eye_in_hand_depth'][:, :, 0]
    sid = f'lib_{step:02d}'
    cv2.imwrite(str(OUT/f'{sid}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(OUT/f'{sid}_depth.npy', dep.astype(np.float32))
    # camera pose in world coordinates, exact from the sim
    cpos = env.sim.data.cam_xpos[cam_id].copy()
    cmat = env.sim.data.cam_xmat[cam_id].copy().reshape(3,3)
    recs.append({'sid': sid, 'cam_pos': cpos.tolist(), 'cam_xmat': cmat.tolist(),
                 'eef_pos': obs['robot0_eef_pos'].tolist(),
                 'dmin': float(dep.min()), 'dmax': float(dep.max()),
                 'dmed': float(np.median(dep))})
    print(f'  {sid}: d {dep.min():.3f}-{dep.max():.3f} med {np.median(dep):.3f} '
          f'campos {np.round(cpos,2)}', flush=True)

meta = {'frames': recs, 'fovy': fovy, 'H': 256, 'W': 256,
        'fx': 256/2/np.tan(np.deg2rad(fovy/2))}
(OUT/'meta.json').write_text(json.dumps(meta, indent=1))
print('fx', round(meta['fx'],1))
env.close()
