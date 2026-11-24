import json,numpy as np
base=json.load(open('mvrecon/out/baseline_rescored.json'))
ag=json.load(open('mvrecon/out/agent_rescored.json'))
print(f"{'method':<16}{'n':>4}{'rot_med':>10}{'trans_med':>11}{'scale_med':>11}{'ov_med':>8}{'<20':>7}")
for name,data,key in [('identity',base,'method'),('A_icp',base,'method'),('scripted_loop',ag,None)]:
 rows=[x for x in data if (x.get(key)==name if key else True)]
 if not rows: continue
 scores=[x['score_corrected'] for x in rows]
 rot=np.array([x['rot_err_deg'] for x in scores]); trans=np.array([x['trans_err_m'] for x in scores]); scale=np.array([x['scale_err'] for x in scores])
 ov=np.array([x['overlap'] if x.get('overlap') is not None else x['score_corrected'].get('overlap_final',np.nan) for x in rows])
 print(f"{name:<16}{len(rows):>4}{np.median(rot):>10.1f}{np.median(trans):>11.3f}{np.median(scale):>11.3f}{np.nanmedian(ov):>8.3f}{int((rot<20).sum()):>4}/{len(rot):<3}")
