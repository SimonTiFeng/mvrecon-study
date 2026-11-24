#!/usr/bin/env python
"""Assemble a self-contained, shareable workbench package.

Contains agent-visible data only (RGB + own-view depth + K for the views that appear in the
task), the frozen CLI, the 40 initial session states, the agent's scored results, the exported
clouds, and the report.  No eval_data / T_true / D_injected is copied.
"""
from __future__ import annotations

import os
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / 'out' / 'agent_study_20261123' / 'package' / 'mvrecon-workbench'
OUT = PKG.parent.parent
STUDY = ROOT / 'study'
SB = pathlib.Path(os.environ.get('MVREBENCH_SANDBOX',
                                 pathlib.Path(os.environ.get('TEMP', '/tmp')) / 'reg_wb_study_20261123')) / 'sessions'


def set_pkg(p):
    global PKG, OUT
    PKG = pathlib.Path(p)
    OUT = PKG.parent.parent


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    (PKG / 'workbench').mkdir(parents=True)
    (PKG / 'data').mkdir()
    (PKG / 'sessions').mkdir()
    (PKG / 'results').mkdir()
    (PKG / 'docs').mkdir()
    (PKG / 'docs' / 'prompts').mkdir()
    for p in (STUDY / 'prompts').glob('*.txt'):
        shutil.copy2(p, PKG / 'docs' / 'prompts' / p.name)
    shutil.copy2(STUDY / 'workbench_cli.py', PKG / 'workbench' / 'workbench_cli.py')
    run = PKG / 'run.py'
    cli = (PKG / 'workbench' / 'workbench_cli.py').read_text(encoding='utf-8')
    old = "SESS = pathlib.Path(os.environ.get('MVRE_SESSIONS', ROOT / 'sessions'))"
    new = ("SESS = pathlib.Path(os.environ['MVRE_SESSIONS']).resolve() "
           "if os.environ.get('MVRE_SESSIONS') else ROOT / 'sessions'")
    assert old in cli, old
    run.write_text(cli.replace(old, new), encoding='utf-8')
    shutil.copy2(PKG / 'workbench' / 'workbench_cli.py',
                 PKG / 'workbench' / 'workbench_cli_frozen.py')
    shutil.copy2(STUDY / 'export_clouds.py', PKG / 'workbench' / 'export_clouds.py')
    sessions = [p for p in SB.iterdir() if (p / 'session.json').exists()]
    sess_data = {p.name: json.loads((p / 'session.json').read_text()) for p in sessions}
    meta = {'scenes': {}}
    used_by_scene = {}
    for name, d in sess_data.items():
        used_by_scene.setdefault(d['scene'], set()).update([d['ref'], d['target']])
    for scene, used in sorted(used_by_scene.items()):
        scene_dir = ROOT / 'agent_data' / scene
        dst = PKG / 'data' / scene
        dst.mkdir()
        src_meta = json.loads((scene_dir / 'meta.json').read_text())
        keep = [im for im in src_meta['images'] if im['sid'] in used]
        for im in keep:
            shutil.copy2(scene_dir / im['depth_file'], dst / im['depth_file'])
            shutil.copy2(scene_dir / im['rgb_file'], dst / im['rgb_file'])
        (dst / 'meta.json').write_text(json.dumps({'scene': scene, 'images': keep},
                                                  indent=1))
        meta['scenes'][scene] = sorted(used)
    for name, s in sess_data.items():
        s['status'] = 'running'
        s['reason'] = ''
        s['used'] = {k: 0 for k in s['budget']}
        s['step'] = 0
        s['history'] = []
        s['last_images'] = []
        s['matches'] = None
        d = PKG / 'sessions' / name
        d.mkdir()
        (d / 'session.json').write_text(json.dumps(s, indent=1))
        (d / 'trace.jsonl').write_text('')
        shutil.copy2(STUDY / 'prompts' / (name + '.txt'), d / 'TASK.md')
    for name in ('baselines.json', 'agent_results.json', 'summary.json', 'per_sample.csv',
                 'stats_tests.json', 'attribution.json', 'attribution_summary.json',
                 'reconstruction_check.json', 'leakage_audit.json'):
        src = OUT / 'eval' / name
        if not src.exists():
            continue
        if name.endswith('.csv'):
            shutil.copy2(src, PKG / 'results' / name)
            continue
        rows = json.loads(src.read_text())
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict):
                    for k in ('matrix_gt', 'T_true', 'D_injected', 'initial_matrix'):
                        r.pop(k, None)
        (PKG / 'results' / name).write_text(json.dumps(rows, indent=1))
    shutil.copy2(OUT / 'report' / 'SUMMARY.md', PKG / 'results' / 'SUMMARY.md')
    shutil.copy2(OUT / 'report' / 'SUMMARY.md', PKG / 'SUMMARY.md')
    shutil.copy2(OUT / 'raw' / 'RUN_MANIFEST.md', PKG / 'docs' / 'RUN_MANIFEST.md')
    shutil.copy2(OUT / 'raw' / 'system_prompt_explore.txt', PKG / 'docs' / 'system_prompt.txt')
    (PKG / 'results' / 'run_meta.json').write_text(
        (OUT / 'raw' / 'run_meta.json').read_text())
    shutil.copy2(STUDY / 'baselines_full.py', PKG / 'workbench' / 'baselines_full.py')
    shutil.copy2(STUDY / 'prepare_study.py', PKG / 'workbench' / 'prepare_study.py')
    shutil.copy2(STUDY / 'README.md', PKG / 'workbench' / 'STUDY_README.md')
    shutil.copy2(STUDY / 'package_README.md', PKG / 'README.md')

    n_sess = sum(1 for _ in (PKG / 'sessions').iterdir())
    total = sum(f.stat().st_size for f in PKG.rglob('*') if f.is_file())
    idx = {'n_sessions': n_sess, 'scenes': meta['scenes'],
           'cli_sha256': hashlib.sha256((PKG / 'workbench' / 'workbench_cli.py').read_bytes())
           .hexdigest(), 'total_bytes': total}
    (PKG / 'PACKAGE_MANIFEST.json').write_text(json.dumps(idx, indent=1))
    print('sessions', n_sess, 'total MB %.1f' % (total / 1e6))
    print('scenes', {k: v for k, v in meta['scenes'].items()})


if __name__ == '__main__':
    main()
