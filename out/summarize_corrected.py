import json,pathlib,numpy as np
p=pathlib.Path(__file__).with_name('baseline_corrected.json')
r=json.loads(p.read_text())
print(f"{'method':<12}{'n':>5}{'rot med':>10}{'trans med':>11}{'scale med':>11}{'ov med':>9}{'<20deg':>9}")
for m in ('identity','A_icp'):
 a=[x for x in r if x['method']==m]
 print(f"{m:<12}{len(a):>5}{np.median([x['rot_err'] for x in a]):>10.1f}"
       f"{np.median([x['trans_err'] for x in a]):>11.3f}{np.median([x['scale_err'] for x in a]):>11.3f}"
       f"{np.median([x['overlap'] for x in a]):>9.3f}{sum(x['rot_err']<20 for x in a):>5}/{len(a):<4}")
print('\nper scene/method:')
for sc in sorted({x['scene'] for x in r}):
 for m in ('identity','A_icp'):
  a=[x for x in r if x['scene']==sc and x['method']==m]
  if a: print(f"{sc:<15}{m:<10}n={len(a):2d} rot={np.median([x['rot_err'] for x in a]):6.1f} trans={np.median([x['trans_err'] for x in a]):6.3f} ov={np.median([x['overlap'] for x in a]):.3f} <20={sum(y['rot_err']<20 for y in a)}/{len(a)}")
print('\noffice v2 original critique example, now current task:')
for x in r:
 if x['scene']=='office' and x['tag']=='v2' and x['method']=='identity':
  print(x['sid'],'rot',x['rot_err'],'trans',x['trans_err'],'D',np.round(np.array(x['matrix_est'])[:3,3],3),'T',np.round(np.array(x['matrix_gt'])[:3,3],3))
