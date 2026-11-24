#!/usr/bin/env python
"""Generate the study report from eval artifacts."""
from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'out' / 'agent_study_20261123'
REPORT = OUT / 'report'


def load(name, default=None):
    p = OUT / 'eval' / name
    return json.loads(p.read_text()) if p.exists() else default


def fmt(m):
    if not m:
        return '—'
    return ('%d | %.2f | %d/%d | %.2f | %d/%d | %d/%d' %
            (m['n'], m['rot_median'], m['rot_lt2'], m['n'],
             m['lat_median'], m['lat_lt0_5'], m['n'],
             m['success'], m['n']))


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    s = load('summary.json')
    st = load('stats_tests.json', {})
    at = load('attribution_summary.json', {})
    at_rows = load('attribution.json', [])
    recon = load('reconstruction_check.json', [])
    groups = {}
    for a in at_rows:
        groups.setdefault(a['case'], a)
    L = []
    L.append('# MV-Recon interactive registration study — full-set report\n')
    L.append('**Date:** 2026-11-23  \n'
             '**Task set:** 40 target instances, 19 unique view pairs, nested v2/v3/v4  \n'
             '**Agent model:** Claude Sonnet 5, one fresh session per case  \n'
             '**Protocol:** initial injected displacement D; score final cumulative M against T_true; '
             'success = rotation < 2° and lateral translation < 0.5 m.\n')
    L.append('\n## 1. What was run\n')
    L.append('The interactive workbench exposed only agent-visible data (two RGB views, each view’s '
             'own predicted depth, intrinsics, current point-cloud state) and generic tools: '
             'region-selecting SIFT/ORB/AKAZE matching, translation / rigid-3D / small-rotation / '
             'reprojection solvers, point-to-plane ICP, FPFH+RANSAC+ICP, manual rotate/translate, '
             'masking, undo and abstain. Budget 30 charged operations per case.\n')
    L.append('Each case was a separate subagent with a fresh context. The harness was audited: the '
             'subagent transcripts contain no project memory index, no parent user prompt, and no '
             '`eval_data` / `T_true` access. Ground truth was read only after the agent finished.\n')
    L.append('\n## 2. Headline comparison (all 40 instances)\n')
    L.append('| method | n | rot median ° | rot <2° | lat median m | lat <0.5 m | success |')
    L.append('|---|---|---|---|---|---|---|')
    for m, v in (s or {}).get('overall', {}).items():
        L.append('| %s | %s |' % (m, fmt(v)))
    L.append('\nNo method was run only on a GT-selected subset. The coarse-rotation “<20°” view is '
             'reported separately in §4 using the explicit rule “A_fpfh_icp rotation <20°”.\n')

    L.append('\n## 3. Per-scene and per-subset\n')
    for m in ('A_fpfh_icp', 'B_sift_rigid', 'C_sift_rigid_sky',
              'G_rotA_sift_reproj', 'H_sift_rigid_reproj', 'AGENT'):
        if m not in (s or {}).get('by_scene', {}):
            continue
        L.append('\n**%s**\n' % m)
        L.append('| scene | n | rot med | <2° | lat med | <0.5 m | success |')
        L.append('|---|---|---|---|---|---|---|')
        for scene, v in s['by_scene'][m].items():
            L.append('| %s | %s |' % (scene, fmt(v)))
        L.append('\n| tag | n | rot med | <2° | lat med | <0.5 m | success |')
        L.append('|---|---|---|---|---|---|---|')
        for tag, v in s['by_tag'][m].items():
            L.append('| %s | %s |' % (tag, fmt(v)))

    L.append('\n## 4. Unique pairs and the GT-conditioned subset\n')
    L.append('The 40 instances repeat 19 scene/view pairs; D differs between nested instances, so '
             'their results are not copies. Deduplicated by taking the lowest subset tag per pair:\n')
    L.append('| method | n | rot median | <2° | lat median | <0.5 m | success |')
    L.append('|---|---|---|---|---|---|---|')
    for m, v in (s or {}).get('unique_pairs', {}).items():
        L.append('| %s | %s |' % (m, fmt(v)))
    sub = (s or {}).get('subset_coarseA_lt20', {})
    L.append('\n**Reporting subset:** %s (n=%d). This is a *reporting* split, not a run filter.\n'
             % (sub.get('rule', ''), sub.get('n', 0)))
    L.append('| method | n | rot median | <2° | lat median | <0.5 m | success |')
    L.append('|---|---|---|---|---|---|---|')
    for m, v in sub.get('methods', {}).items():
        L.append('| %s | %s |' % (m, fmt(v)))

    L.append('\n## 5. Agent behaviour\n')
    L.append('- Instances finished: %d; abstained: %d; no finish: %d.'
             % (s.get('n_done', 0), s.get('n_abstain', 0), s.get('n_no_finish', 0)))
    L.append('- Trajectory reconstruction passed: %d/%d (max replay error %.2e).'
             % (s.get('reconstruction_passed', 0), s.get('n_instances', 0),
                max([r['final_replay_error'] for r in recon], default=0.0)))
    L.append('- Median charged operations: %s of 30; median trace actions: %s.'
             % (s.get('agent_budget_used_median'), s.get('agent_actions_median')))
    L.append('- Totals: match %s, solve %s, ICP %s, FPFH %s, undo %s.'
             % (s.get('agent_match_total'), s.get('agent_solve_total'),
                s.get('agent_icp_total'), s.get('agent_fpfh_total'),
                s.get('agent_undo_total')))
    if at:
        L.append('- State-change sequences: %s.' % json.dumps(at.get('sequence_groups', {})))
        L.append('- Final transform identical (≤0.1°, ≤1 cm) to a traditional baseline in %d/%d cases; '
                 'closest baseline counts: %s.'
                 % (at.get('n_closest_same', 0), at.get('n', 0),
                    json.dumps(at.get('closest_counts', {}))))

    L.append('\n## 6. Paired tests vs each baseline\n')
    L.append('Wilcoxon signed-rank on the 40 paired errors; success by exact McNemar.\n')
    L.append('| baseline | rot Δ median | p | lat Δ median | p | success A/B | McNemar p |')
    L.append('|---|---|---|---|---|---|---|')
    for m, c in (st or {}).get('comparisons', {}).items():
        L.append('| %s | %+.3f | %.3g | %+.3f | %.3g | %d/%d | %.3g |' %
                 (m, c['rot_delta_median'], c['rot_wilcoxon_p'] or 1,
                  c['lat_delta_median'], c['lat_wilcoxon_p'] or 1,
                  c['agent_successes'], c['baseline_successes'],
                  c['mcnemar_exact_p'] or 1))

    # per-case deltas vs H
    base = load('baselines.json', [])
    agent = load('agent_results.json', [])
    h = {('%s_%s_%s' % (r['scene'], r['tag'], r['sid'])): r
         for r in base if r['method'] == 'H_sift_rigid_reproj' and 'rot_err' in r}
    L.append('\n## 7. Cases where the agent differs from the strongest baseline\n')
    L.append('Δ is agent minus H_sift_rigid_reproj (negative is better).\n')
    L.append('| case | status | agent rot / lat | H rot / lat | Δlat | sequence |')
    L.append('|---|---|---|---|---|---|')
    for a in sorted(agent, key=lambda r: r['lat_err'] - h.get(r['case'], {}).get('lat_err', 0)):
        hr = h.get(a['case'])
        if not hr:
            continue
        dl = a['lat_err'] - hr['lat_err']
        if abs(dl) < 0.02 and abs(a['rot_err'] - hr['rot_err']) < 0.5:
            continue
        g = groups.get(a['case'], {})
        L.append('| %s | %s | %.2f° / %.2f m | %.2f° / %.2f m | %+.2f | %s |' %
                 (a['case'], a['status'], a['rot_err'], a['lat_err'],
                  hr['rot_err'], hr['lat_err'], dl,
                  ', '.join(g.get('sequence', [])) or 'none'))

    L.append('\n## 8. Direct answers\n')
    h_succ = ((s or {}).get('overall', {}).get('H_sift_rigid_reproj', {}) or {}).get('success')
    a_succ = ((s or {}).get('overall', {}).get('AGENT', {}) or {}).get('success')
    agent_only = [r for r in agent if r.get('success') and
                  not h.get(r['case'], {}).get('success')]
    h_only = [r for r in agent if (not r.get('success')) and
              h.get(r['case'], {}).get('success')]
    L.append('- **Which operations did the agent choose?** %s. The dominant sequence was '
             'SIFT match → rigid-3D RANSAC → ICP; FPFH was used when SIFT failed, and '
             'reprojection was used, accepted, or undone case by case.'
             % json.dumps((at or {}).get('sequence_groups', {})))
    L.append('- **Where did it improve or degrade?** It crossed the success threshold on %d cases '
             'where H failed; H crossed it on %d cases where the agent failed. Success counts: '
             'agent %s, H %s (of 40).'
             % (len(agent_only), len(h_only), a_succ, h_succ))
    if agent_only:
        L.append('  - agent-only successes: %s.' % ', '.join(r['case'] for r in agent_only))
    if h_only:
        L.append('  - H-only successes: %s.' % ', '.join(r['case'] for r in h_only))
    L.append('- **Was there a real gain over the strongest baseline?** On the full set, no: the '
             'agent is competitive on median lateral error but converts fewer cases into joint '
             'rotation+lateral successes, and it uses 5–20× the operations of the fixed pipeline. '
             'It did avoid H’s catastrophic failures on facade_v4_0201 (7.66°/1.49 m vs '
             '73.93°/31.72 m) and improved both facade 0209 instances, but rotation stayed above '
             'the 2° threshold there, so these are error reductions, not success flips.')
    L.append('- **Did any gain come from multi-round judgment rather than the tools?** The '
             'numerical work is done by SIFT/FPFH/ICP/reprojection. Against its own first solver '
             'output, later rounds improved rotation in %d and worsened it in %d cases; improved '
             'lateral in %d and worsened it in %d. The final transform is identical to a '
             'traditional baseline in %d of %d cases. The model’s measurable contribution is '
             'tool selection and rejecting bad refinements, not better numerics.'
             % ((at or {}).get('multi_round', {}).get('rot_improved', 0),
                (at or {}).get('multi_round', {}).get('rot_worsened', 0),
                (at or {}).get('multi_round', {}).get('lat_improved', 0),
                (at or {}).get('multi_round', {}).get('lat_worsened', 0),
                (at or {}).get('n_closest_same', 0), (at or {}).get('n', 0)))
    L.append('- **Unfinished items, leakage risk and limitations.** %d abstention(s). The run had '
             'no GT in the agent context; the audit found no memory index, parent prompt, '
             'eval_data or T_true access in any transcript. One of 40 transcripts contains an '
             'inline numpy matrix-arithmetic command outside the workbench CLI (no GT, no '
             'external data); its final state is still the workbench state and is reported, with '
             'and without it unchanged in conclusion. Residual risks: the subagent’s cwd is the '
             'project directory (its name is visible, though no project file was read), one sample '
             'per case (no model-stochasticity average), and depth noise dominates several scenes. '
             'Interrupted agents were resumed from their own workbench state; cases with no finish '
             'are reported as no_finish, not dropped.'
             % (s or {}).get('n_abstain', 0))

    L.append('\n## 9. Evidence and limitations\n')
    L.append('- Raw: `raw/transcripts/`, `raw/final_reports/`, `raw/launches/`; workbench traces and '
             'state: the sandbox `sessions/`; scored data: `eval/`.')
    L.append('- The workbench enforces budgets, serialises parallel tool calls with a per-session lock, '
             'and logs every operation with before/after matrices; final transforms replay exactly.')
    L.append('- Two harness defects were found and fixed before the reported run: a lost-update race '
             'between parallel CLI calls, and a `closeup` JSON serialisation error. All reported sessions '
             'use one frozen CLI (hash in `study/sandbox_manifest.json`).')
    L.append('- The Agent-tool subagents run with cwd in the project directory; their system context '
             'contains the directory name but not the memory index or the parent prompt. Any Bash command '
             'other than the workbench CLI is flagged by `eval/leakage_audit.json`.')
    L.append('- The agent may decline; abstentions are scored on the final state and reported, not dropped.')
    L.append('- No repeated runs per case: model stochasticity is not averaged. This is a single '
             'sample of 40 independent sessions, not a confidence interval on the model.')
    (REPORT / 'SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')
    print('wrote', REPORT / 'SUMMARY.md')


if __name__ == '__main__':
    main()
