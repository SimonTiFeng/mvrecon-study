import json, pathlib, numpy as np
root=pathlib.Path('mvrecon'); oldroot=root/'archive/pre_pose_fix'
oldtask=json.load(open(oldroot/'task.json')); oldres=json.load(open(oldroot/'baseline_table.json'))
poses={x['sid']:x for x in json.load(open(root/'eval_data/office/meta.json'))['images']}
def rd(R):
 c=(np.trace(R)-1)/2
 return float(np.degrees(np.arccos(np.clip(c,-1,1))))
# legacy office v2
lo=oldtask['office']['v2']; sid=lo['ids'][1]; ref=lo['ref']
D=np.asarray(lo['D_injected'][sid]); Tb=np.asarray(lo['T_true'][sid]); Ab=Tb@np.linalg.inv(D)
legacy_score=rd(D[:3,:3]@Ab[:3,:3].T)
legacy_direct=rd(D[:3,:3]@Tb[:3,:3].T)
Ri,ti=np.asarray(poses[ref]['R']),np.asarray(poses[ref]['t'])
Rj,tj=np.asarray(poses[sid]['R']),np.asarray(poses[sid]['t'])
Tg=np.eye(4); Tg[:3,:3]=Ri@Rj.T; Tg[:3,3]=ti-Ri@Rj.T@tj
correct_old_D_vs_T=rd(D[:3,:3]@Tg[:3,:3].T)
saved=[x['rot_err'] for x in oldres if x.get('scene')=='office' and x.get('tag')=='v2' and x.get('method')=='identity']
# current task/results check
curtask=json.load(open(root/'eval_data/task.json'))
base=json.load(open(root/'out/baseline_rescored.json')); agent=json.load(open(root/'out/agent_rescored.json'))
bad=[]
for x in base:
 t=curtask[x['scene']][x['tag']]
 if x['sid'] not in t['ids'][1:]: bad.append('baseline id')
 if np.max(np.abs(np.asarray(x['matrix_gt'])-np.asarray(t['T_true'][x['sid']])))>1e-8: bad.append('baseline T')
 if x['method']=='identity' and np.max(np.abs(np.asarray(x['matrix_est'])-np.asarray(t['D_injected'][x['sid']])))>1e-8: bad.append('identity D')
for x in agent:
 t=curtask[x['scene']][x['tag']]
 if x['sid'] not in t['ids'][1:]: bad.append('agent id')
 if np.max(np.abs(np.asarray(x['T_true_eval_only'])-np.asarray(t['T_true'][x['sid']])))>1e-8: bad.append('agent T')
# summaries
summ={}
for name,rows,fn in [('identity',[x for x in base if x['method']=='identity'],lambda x:x),
                      ('FPFH+ICP',[x for x in base if x['method']=='A_icp'],lambda x:x['score_corrected']),
                      ('scripted_loop',agent,lambda x:x['score_corrected'])]:
 a=[fn(x) for x in rows]
 rot=np.array([x.get('rot_err_deg',x.get('rot_err')) for x in a]); tr=np.array([x.get('trans_err_m',x.get('trans_err')) for x in a])
 summ[name]={'n':len(a),'rot_median':float(np.median(rot)),'trans_median_m':float(np.median(tr)),
             'rot_lt20':int((rot<20).sum()),'trans_lt05':int((tr<.5).sum()),
             'joint_rot_lt20_trans_lt05':int(((rot<20)&(tr<.5)).sum())}
report={
 'legacy_office_v2':{'pair':[ref,sid],'saved_result_rot_deg':saved[0] if saved else None,
     'legacy_scorer_M_vs_A_rot_deg':legacy_score,
     'M_vs_old_T_bad_rot_deg':legacy_direct,
     'M_vs_correct_COLMAP_T_rot_deg':correct_old_D_vs_T,
     'note':'The saved 83.4 does not reproduce from the archived task using either the legacy or corrected formula; baseline_table/task provenance mismatch.'},
 'current_office_v2':{'pair':curtask['office']['v2']['ids'],
     'identity_initial_rot_deg':next(x['score_corrected']['rot_err_deg'] for x in base if x['scene']=='office' and x['tag']=='v2' and x['method']=='identity')},
 'formula':{'legacy':'score cumulative M against incremental A=T_true@inv(D)',
            'correct':'score cumulative M against T_true; after applying A, M=A@D=T_true'},
 'current_task_transforms_verified':len(base)==80 and len(agent)==40 and not bad,
 'current_task_mismatches':bad,
 'corrected_summary':summ,
 'saved_corrected_rows':{'baseline':len(base),'agent':len(agent),
   'baseline_has_initial_D_final_M_T_and_method_matrices':all('initial_matrix'in x and 'matrix_est'in x and 'matrix_gt'in x and 'method_result'in x for x in base),
   'agent_has_initial_final_T_D_full_trace':all('initial_matrix'in x and 'final_matrix'in x and 'T_true_eval_only'in x and 'D_injected_eval_only'in x and 'trace'in x and 'hist'in x for x in agent)}}
out=root/'out'/'correction_audit.json';out.write_text(json.dumps(report,indent=1))
print(json.dumps(report,indent=1))
