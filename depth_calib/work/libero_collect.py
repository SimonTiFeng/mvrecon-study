"""Collect LIBERO wrist-camera frames with exact ground-truth depth and camera pose.

The metric fact here is the SIMULATOR: every depth is exact, every pose is exact.
This is the cleanest possible test bed for the method — no SfM, no sensor noise.
"""
import sys, os, json, pathlib
sys.path.insert(0, r'D:/libero_src')
os.environ.setdefault('MUJOCO_GL','wgl')
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

BDDL = r'D:/libero_src/libero/libero/bddl_files/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl'
OUT = pathlib.Path('D:/libero_out'); OUT.mkdir(exist_ok=True)

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256,
                         camera_names=['agentview','robot0_eye_in_hand'], camera_depths=True)
env.reset()
print('env ready', flush=True)

recs = []
for step in range(6):
    act = np.zeros(7)
    if step > 0:
        act[0] = 0.3  # small forward motion so frames differ
    obs, _, _, _ = env.step(act)
    rgb = obs['robot0_eye_in_hand_image']
    dep = obs['robot0_eye_in_hand_depth'][:, :, 0]
    sid = f'libero_{step:02d}'
    pathlib.Path(OUT / f'{sid}.png').write_bytes(
        __import__('cv2').imencode('.png', __import__('cv2').cvtColor(
            rgb.astype(np.uint8), __import__('cv2').COLOR_RGB2BGR))[1])
    np.save(OUT / f'{sid}_depth.npy', dep.astype(np.float32))
    recs.append({'sid': sid, 'eef_pos': obs['robot0_eef_pos'].tolist(),
                 'eef_quat': obs['robot0_eef_quat'].tolist(),
                 'depth_min': float(dep.min()), 'depth_max': float(dep.max()),
                 'depth_med': float(np.median(dep))})
    print(f'  {sid}: depth {dep.min():.3f}-{dep.max():.3f} med {np.median(dep):.3f}', flush=True)

# camera extrinsics from the sim
cam = env.sim.model.camera_name2id('robot0_eye_in_hand')
pos = env.sim.model.cam_pos[cam]
quat = env.sim.model.cam_quat[cam]
fovy = float(env.sim.model.cam_fovy[cam])
meta = {'frames': recs, 'cam_pos': pos.tolist(), 'cam_quat': quat.tolist(),
        'fovy': fovy, 'H': 256, 'W': 256,
        'fx': 256/2/np.tan(np.deg2rad(fovy/2))}
pathlib.Path(OUT / 'meta.json').write_text(json.dumps(meta, indent=1))
print('camera pos', np.round(pos,3), 'fovy', fovy, 'fx', round(meta['fx'],1))
env.close()
