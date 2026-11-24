"""Run identity baseline on the corrected task and persist full matrices."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import evaluate, method_identity

ROOT = HERE.parent
TASK = json.loads((ROOT / 'eval_data/task.json').read_text())
out = []
for scene in sorted(TASK):
    for tag in ('v2','v3','v4'):
        rows = evaluate(scene, tag, method_identity, verbose=False)
        for r in rows:
            r.update({'scene':scene,'tag':tag,'method':'identity'})
        out.extend(rows)
        print(scene, tag, len(rows), flush=True)
p = ROOT/'out'/'identity_corrected.json'
p.write_text(json.dumps(out,indent=1))
print('wrote',p,'rows',len(out))
