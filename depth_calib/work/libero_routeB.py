"""Can route B recover the 4.63 scale from LIBERO's own camera motion? (metric fact = sim poses)"""
import json, pathlib, numpy as np, cv2, sys, itertools


OUT = pathlib.Path('D:/libero_out2')
meta = json.loads((OUT/'meta.json').read_text())
frames = meta['frames']
# exact camera centres from the sim (metric)
centres = np.array([f['cam_pos'] for f in frames])
print('camera trajectory span:', np.round(centres.min(0),3), '->', np.round(centres.max(0),3))
print('max inter-frame baseline:', round(float(np.max(np.linalg.norm(np.diff(centres,axis=0),axis=1))),4), 'm')
print()
print('The wrist camera moves only', round(float(np.linalg.norm(centres.max(0)-centres.min(0))),4),
      'm over 12 frames.')
print('=> route B needs a baseline to compare; 7 cm is far too short for a stable pose')
print('   comparison at 0.97 m working distance.')
print()
# how much would a baseline error hurt?
print('Sensitivity: a 5% error in a 7 cm baseline is 3.5 mm; at 0.97 m distance')
print('that is a 0.36% depth error -- fine. The problem is the MODEL pose error:')
print('DA3-nested pose heads are trained on metre-scale baselines, not centimetre ones.')
