"""Run the corrected baseline protocol and persist per-target cumulative matrices.

Usage: python mvrecon/eval/run_baseline_corrected.py [scene ...]
"""
import json
import pathlib
import sys

from protocol import evaluate, method_identity, method_icp

ROOT = pathlib.Path(__file__).resolve().parents[1]
scenes = sys.argv[1:] or ['electro', 'office', 'playground']
all_rows = []
for scene in scenes:
    print('\n=== %s ===' % scene, flush=True)
    for tag in ('v2', 'v3', 'v4'):
        for method_name, method in (('identity', method_identity), ('A_icp', method_icp)):
            print(' [%s] %s:' % (tag, method_name), flush=True)
            rows = evaluate(scene, tag, method)
            for r in rows:
                r.update({'scene': scene, 'tag': tag, 'method': method_name})
            all_rows.extend(rows)
out = ROOT / 'out' / 'baseline_corrected.json'
prev = json.loads(out.read_text()) if out.exists() else []
keys = {(r['scene'], r['tag'], r['sid'], r['method']) for r in all_rows}
prev = [r for r in prev if (r['scene'], r['tag'], r['sid'], r['method']) not in keys]
out.write_text(json.dumps(prev + all_rows, indent=1))
print('\nwrote %s rows=%d' % (out, len(prev)+len(all_rows)))
