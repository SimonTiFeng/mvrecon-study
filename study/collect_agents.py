#!/usr/bin/env python
"""Collect Agent-tool subagent transcripts into the study's raw evidence tree.

The Agent tool's .output file is a zero-byte placeholder; the real JSONL lives at
~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'out' / 'agent_study_20261123'
# <Agent-tool run dir>/subagents/ — outside the sandbox, so it is given explicitly
SUB = pathlib.Path(os.environ['MVREBENCH_SUBAGENTS'])
LAUNCH = OUT / 'raw' / 'launches'
TRANS = OUT / 'raw' / 'transcripts'
REPORTS = OUT / 'raw' / 'final_reports'


def parse(path):
    texts, tools, bad = [], [], 0
    for line in path.read_text(errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            bad += 1
            continue
        msg = rec.get('message')
        if isinstance(msg, dict) and rec.get('type') == 'assistant':
            for b in msg.get('content') or []:
                if isinstance(b, dict):
                    if b.get('type') == 'text':
                        texts.append(b.get('text', ''))
                    elif b.get('type') == 'tool_use':
                        tools.append({'name': b.get('name'), 'input': b.get('input')})
    final = texts[-1] if texts else ''
    return {'n_text_blocks': len(texts), 'n_tool_uses': len(tools),
            'n_bad_lines': bad, 'final': final,
            'tool_names': [t['name'] for t in tools]}


def main():
    TRANS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in sorted(LAUNCH.glob('batch*.json')):
        batch = json.loads(f.read_text())
        for c in batch.get('cases', []) + ([batch['pilot_case']] if 'pilot_case' in batch else []):
            case, aid = c['case'], c['agent_id']
            src = SUB / ('agent-%s.jsonl' % aid)
            if not src.exists():
                summary.append({'case': case, 'agent_id': aid, 'missing': True})
                continue
            dst = TRANS / (case + '.jsonl')
            shutil.copy2(src, dst)
            info = parse(dst)
            (REPORTS / (case + '.txt')).write_text(info['final'], encoding='utf-8')
            summary.append({'case': case, 'agent_id': aid, 'transcript': str(dst),
                            'size': dst.stat().st_size, **{k: v for k, v in info.items()
                                                           if k != 'final'},
                            'final_chars': len(info['final'])})
    (OUT / 'raw' / 'collect_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
