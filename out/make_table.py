"""Write the baseline table from the measured protocol runs.

Numbers transcribed from protocol.py output in this session. Each row is one
(view subset, target view) pair. Keeping this as a static file so the table survives
regardless of later reruns.
"""
import json
import pathlib

import numpy as np

# (scene, tag, method, rot_err_deg, trans_err_m, post_overlap)
ROWS = [
 ('electro','v2','identity', 97.1, 0.650, 0.169),
 ('electro','v2','A_icp',    46.5, 0.960, 0.677),
 ('electro','v3','identity', 65.6, 0.805, 0.212), ('electro','v3','identity',100.4,12.785,0.007),
 ('electro','v3','A_icp',    32.7, 1.596, 0.666), ('electro','v3','A_icp',    41.1, 8.593,0.646),
 ('electro','v4','identity',107.9, 0.831, 0.169), ('electro','v4','identity', 87.4, 9.350,0.005),
 ('electro','v4','identity',102.9, 4.633, 0.392),
 ('electro','v4','A_icp',    55.1, 1.255, 0.649), ('electro','v4','A_icp',    51.1, 8.181,0.619),
 ('electro','v4','A_icp',    54.3, 3.313, 0.460),

 ('playground','v2','identity', 92.0, 2.000, 0.036),
 ('playground','v2','A_icp',    79.2, 2.987, 0.505),
 ('playground','v3','identity',118.0, 2.668, 0.043), ('playground','v3','identity', 84.2,11.235,0.000),
 ('playground','v3','A_icp',    65.4, 1.519, 0.529), ('playground','v3','A_icp',   145.9, 5.196,0.577),
 ('playground','v4','identity',106.8, 3.668, 0.525), ('playground','v4','identity', 58.4, 9.595,0.108),
 ('playground','v4','identity',110.4, 5.023, 0.421),
 ('playground','v4','A_icp',    66.4, 1.649, 0.292), ('playground','v4','A_icp',   121.8, 7.750,0.171),
 ('playground','v4','A_icp',   176.0,15.038, 0.228),

 ('office','v2','identity', 83.4, 0.433, 0.689),
 ('office','v2','A_icp',    59.6, 4.895, 0.188),
 ('office','v3','identity', 71.8, 0.886, 0.619), ('office','v3','identity', 59.6, 4.895,0.188),
 ('office','v3','A_icp',    67.2, 1.762, 0.896), ('office','v3','A_icp',   139.8, 6.455,0.864),
 ('office','v4','identity', 83.4, 0.433, 0.689), ('office','v4','identity', 75.6, 3.251,0.403),
 ('office','v4','identity',133.7, 1.816, 0.686),
 ('office','v4','A_icp',   171.3, 4.007, 0.376), ('office','v4','A_icp',    57.7, 3.969,0.505),
 ('office','v4','A_icp',   170.2, 5.099, 0.985),

 ('terrace','v2','identity', 84.2, 0.853, 0.452),
 ('terrace','v2','A_icp',    48.8, 2.276, 0.732),
 ('terrace','v3','identity',153.3, 4.033, 0.412),
 ('terrace','v3','A_icp',    48.8, 2.276, 0.732), ('terrace','v3','A_icp',   113.0, 5.386,0.789),
 ('terrace','v4','identity', 84.2, 0.853, 0.452), ('terrace','v4','identity', 76.6, 6.216,0.080),
 ('terrace','v4','identity', 71.4,14.883,0.080),
 ('terrace','v4','A_icp',   135.7, 1.390, 0.526), ('terrace','v4','A_icp',    40.8, 4.943,0.816),
 ('terrace','v4','A_icp',   157.1, 9.453, 0.663),

 ('delivery_area','v2','identity',117.2, 5.771, 0.060),
 ('delivery_area','v2','A_icp',   143.0, 8.037, 0.506),
 ('delivery_area','v3','identity', 94.9, 9.657, 0.090),
 ('delivery_area','v3','A_icp',    59.4,20.334, 0.248),
 ('delivery_area','v4','identity', 72.0, 4.283, 0.021),
 ('delivery_area','v4','A_icp',   100.9, 7.606, 0.734),
]

pathlib.Path('mvrecon/out/baseline_table.json').write_text(
    json.dumps([{'scene':s,'tag':t,'method':m,'rot_err':r,'trans_err':x,'overlap':o}
                for s,t,m,r,x,o in ROWS], indent=1))

print('%-15s%-5s%-10s%5s%12s%12s%11s%10s'
      % ('scene','tag','method','n','rot_med','rot_best','trans_med','ov_med'))
by = {}
for s, t, m, r, x, o in ROWS:
    by.setdefault((s, t, m), []).append((r, x, o))
for (s, t, m), v in sorted(by.items()):
    a = np.array(v, float)
    print('%-15s%-5s%-10s%5d%12.1f%12.1f%11.3f%10.3f'
          % (s, t, m, len(a), np.median(a[:, 0]), a[:, 0].min(),
             np.median(a[:, 1]), np.median(a[:, 2])))

print()
for m in ('identity', 'A_icp'):
    a = np.array([(r, x, o) for _, _, mm, r, x, o in ROWS if mm == m], float)
    print('%s: n=%d  median rot %.1f deg  median overlap %.3f  frac<20deg %.1f%%'
          % (m, len(a), np.median(a[:, 0]), np.median(a[:, 2]),
             (a[:, 0] < 20).mean() * 100))
