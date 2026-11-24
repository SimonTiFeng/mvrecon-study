#!/usr/bin/env python
"""Build per-case run metadata and save the shared subagent system prompt."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'out' / 'agent_study_20261123'


def main():
    launches = {}
    for f in sorted((OUT / 'raw' / 'launches').glob('batch*.json')):
        b = json.loads(f.read_text())
        for c in b.get('cases', []):
            launches[c['case']] = c['agent_id']
    rows = []
    for case, aid in sorted(launches.items()):
        d = OUT / 'raw' / 'transcripts' / (case + '.jsonl')
        prompt = ROOT / 'study' / 'prompts' / (case + '.txt')
        sess_p = pathlib.Path(os.environ.get('MVREBENCH_SANDBOX',
                                             pathlib.Path(os.environ.get('TEMP', '/tmp')) / 'reg_wb_study_20261123'
                                             )) / 'sessions' / case / 'session.json'
        sess = json.loads(sess_p.read_text()) if sess_p.exists() else {}
        if not sess_p.exists():
            sess['source'] = 'transcript+launches only (sandbox pruned)'
        rows.append({
            'case': case, 'agent_id': aid, 'model': 'claude-sonnet-5',
            'harness': 'Agent tool / subagent_type=Explore',
            'prompt_file': str(prompt),
            'prompt_sha256': (hashlib.sha256(prompt.read_bytes()).hexdigest()
                              if prompt.exists() else None),
            'transcript': str(d) if d.exists() else None,
            'budget': sess.get('budget'), 'used': sess.get('used'),
            'status': sess.get('status'), 'steps': sess.get('step'),
        })
    (OUT / 'raw' / 'run_meta.json').write_text(json.dumps(rows, indent=1))
    # shared system prompt and model identity from the transcript attachments
    sysp = None
    model_id = None
    for r in rows:
        if not r['transcript']:
            continue
        for line in pathlib.Path(r['transcript']).read_text(errors='replace').splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            a = rec.get('attachment')
            if isinstance(a, dict):
                if a.get('type') == 'prompt_snapshot' and a.get('systemPrompt'):
                    sysp = '\n'.join(a['systemPrompt'])
                if isinstance(a.get('identity'), dict):
                    model_id = a['identity'].get('modelId') or model_id
            if sysp and model_id:
                break
        if sysp and model_id:
            break
    if sysp:
        (OUT / 'raw' / 'system_prompt_explore.txt').write_text(sysp, encoding='utf-8')
    if model_id:
        for r in rows:
            r['model'] = model_id
        (OUT / 'raw' / 'run_meta.json').write_text(json.dumps(rows, indent=1))
    print('wrote run_meta for', len(rows), 'cases; system prompt chars',
          len(sysp or ''), 'model', model_id)


if __name__ == '__main__':
    main()
