"""Verify: after injection the clouds are genuinely misaligned, and the true transform restores them."""
import json, pathlib, sys, numpy as np
sys.path.insert(0,'mvrecon/workbench')
from wb import Workbench
AG=pathlib.Path('mvrecon/agent_data')
TASK=json.loads(pathlib.Path('mvrecon/eval_data/task.json').read_text())
print(f"{'scene':<14}{'tag':<4}{'ov_start':>9}{'ov@true':>9}{'ov@identity':>12}")
for scene in sorted(TASK):
    for tag in ('v2','v3','v4'):
        t=TASK[scene][tag]; ids=t['ids']; ref=t['ref']
        wb=Workbench(AG/scene, stride=32)
        wb.set_reference(ref)
        tgt=ids[1]
        # inject
        for sid in ids:
            M=np.asarray(t['D_injected'][sid])
            if sid==ref: continue
            # decompose into R,t,s and set directly
            c=wb.clouds[sid]
            Sc = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
            c.R=M[:3,:3]/Sc; c.t=M[:3,3]; c.s=float(c.s*Sc)
        ov_start=wb.overlap_stats(tgt)[0]
        # The agent sees D p_j and must produce T p_j, so it applies A = T @ inv(D).
        Tt = np.asarray(t['T_true'][tgt]); D = np.asarray(t['D_injected'][tgt])
        want = Tt @ np.linalg.inv(D)
        wb._snap()
        wb.apply(tgt, R=want[:3,:3], t=want[:3,3])
        ov_true=wb.overlap_stats(tgt)[0]
        wb.undo()
        ov_id=wb.overlap_stats(tgt)[0]
        print(f"{scene:<14}{tag:<4}{ov_start:>9.3f}{ov_true:>9.3f}{ov_id:>12.3f}")
