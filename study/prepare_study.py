#!/usr/bin/env python
"""Build the agent sandbox for the interactive-registration study.

The sandbox contains ONLY agent-visible material:
  workbench_cli.py, data/<scene>/{png,npy,meta.json}, sessions/<case>/session.json

No eval_data, no T_true, no thetas.  Data files are hard-linked where possible so the
7 GB of agent_data is not duplicated.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDY = pathlib.Path(__file__).resolve().parent
DEFAULT_SANDBOX = pathlib.Path(os.environ.get('TEMP', '/tmp')) / 'reg_wb_study_20261123'
BUDGET = {'photo': 6, 'render': 12, 'closeup': 6, 'match': 5, 'mask': 4,
          'transform': 8, 'solve': 6, 'icp': 4, 'fpfh': 2, 'total': 30}


def hardlink_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return 'exists'
    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        shutil.copy2(src, dst)
        return 'copy'


def build(sandbox: pathlib.Path):
    task = json.loads((ROOT / 'eval_data' / 'task.json').read_text())
    if sandbox.exists():
        shutil.rmtree(sandbox)
    (sandbox / 'sessions').mkdir(parents=True)
    (sandbox / 'data').mkdir()
    modes = {}
    for scene in sorted(os.listdir(ROOT / 'agent_data')):
        sd = ROOT / 'agent_data' / scene
        if not (sd / 'meta.json').exists():
            continue
        for f in sorted(sd.iterdir()):
            if f.is_file():
                modes[f.name] = hardlink_or_copy(f, sandbox / 'data' / scene / f.name)
    shutil.copy2(STUDY / 'workbench_cli.py', sandbox / 'workbench_cli.py')
    (sandbox / 'workbench_cli.py').chmod(0o755)
    cli_hash = hashlib.sha256((STUDY / 'workbench_cli.py').read_bytes()).hexdigest()[:16]

    cases = []
    for scene, subs in task.items():
        for tag, t in subs.items():
            for sid in t['ids']:
                if sid == t['ref']:
                    continue
                case = '%s_%s_%s' % (scene, tag, sid)
                D = np.asarray(t['D_injected'][sid], float)
                sc = float(np.cbrt(abs(np.linalg.det(D[:3, :3]))))
                sess = {
                    'case': case, 'scene': scene, 'tag': tag,
                    'ref': t['ref'], 'target': sid,
                    'poses': {t['ref']: {'R': np.eye(3).tolist(), 't': [0.0] * 3, 's': 1.0},
                              sid: {'R': (D[:3, :3] / sc).tolist(),
                                    't': D[:3, 3].tolist(), 's': sc}},
                    'initial_matrix': D.tolist(),
                    'mask': {'max_dist': None, 'drop_sky': False},
                    'budget': dict(BUDGET),
                    'used': {k: 0 for k in BUDGET},
                    'step': 0, 'history': [], 'status': 'running', 'reason': '',
                    'last_images': [], 'matches': None,
                    'model': 'claude-sonnet-5',
                }
                d = sandbox / 'sessions' / case
                d.mkdir(parents=True)
                (d / 'session.json').write_text(json.dumps(sess, indent=1))
                (d / 'trace.jsonl').write_text('')
                cases.append({'case': case, 'scene': scene, 'tag': tag,
                              'ref': t['ref'], 'target': sid,
                              'D_injected': D.tolist()})
    (sandbox / 'cases.json').write_text(json.dumps(cases, indent=1))
    meta = {'sandbox': str(sandbox), 'cli_sha256_16': cli_hash,
            'n_cases': len(cases), 'budget': BUDGET,
            'link_modes': {m: sum(1 for v in modes.values() if v == m)
                           for m in set(modes.values())}}
    (STUDY / 'sandbox_manifest.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return cases


def prompt(case, scene, tag, ref, target, cli):
    return f'''你正在操作一个三维配准工作台。

目标：将目标视图（moving）的点云配准到参考视图（reference），使两者中属于同一场景的结构尽可能正确地对齐。

案例信息
- 案例编号：{case}
- 场景：{scene}（子集标签 {tag}）
- 参考视图 reference（固定不动）：{ref}
- 移动视图 moving（只有它会移动）：{target}
- 两幅 RGB 图像都是 6048x4032；所有像素坐标使用原始分辨率。图像上的黄色网格线每 500 px 一条，并标注了坐标。

工作台
在 Bash 中一次运行一条命令：
  python "{cli}" {case} <command> [参数]

每条命令输出 JSON。如果 JSON 中有 "images"，请用 Read 工具打开这些 PNG 查看；这是你观察照片、点云和匹配结果的方式。

可用命令（括号内为每次消耗的操作数）：
- state：当前累计变换、掩码、预算和已保存的对应关系（0）
- stats：当前配准状态的重叠统计（0）
- photos：两张原始 RGB 并排，带坐标网格（1）
- photo ref|target：单张原始 RGB，带坐标网格（1）
- render [iso|top|side|front|back|left]：从指定视角渲染当前点云状态（1）
- closeup ref|target u0 v0 u1 v1：裁剪原始图像的一个区域，并给出该区域当前的三维统计（1）
- match [--matcher sift|orb|akaze] [--ref-box u0 v0 u1 v1] [--mov-box u0 v0 u1 v1]：在两张图之间做特征匹配，保存对应关系，输出匹配数、当前变换下的三维残差、隐含平移散布，并生成匹配可视化（1）
- mask [--max-dist M] [--drop-sky] [--keep-sky] [--clear]：设置后续 stats/solve/icp/fpfh 使用的点过滤（1）
- rotate x|y|z <deg>：绕世界坐标轴旋转移动点云（1）
- translate <dx> <dy> <dz>：平移移动点云，单位米（1）
- solve translation|rigid3d|rotcorr|reproj：用已保存的对应关系求变换（1）
    translation 保持当前旋转，只求平移；
    rigid3d 用 RANSAC 求完整刚体变换；
    rotcorr 在当前旋转附近拟合小旋转修正；
    reproj 最小化参考图像上的重投影误差。
- icp [radius]：从当前状态运行点到面 ICP（1）
- fpfh：从当前状态运行 FPFH+RANSAC+ICP（1）
- undo：撤销上一次状态改变（0）
- finish --status done|abstain --reason "..."：结束并说明判断（0）

预算：总操作 30 次。photo 6、render 12、closeup 6、match 5、mask 4、transform 8、solve 6、icp 4、fpfh 2。
state / stats / undo / finish 不消耗预算。

规则
- 只通过工作台命令观察和操作。不要读取工作台目录之外的文件，不要使用其他配准代码或外部数据。
- 一次运行一条命令，看过结果和图像后再决定下一步。
- 每次操作前用一句话说明你要检查或改变什么。
- 深度是分别预测的，可能有噪声；两张图可能只部分重叠，也可能存在相似但并非同一物体的结构。不要把点云重合度单独当成一定正确的证据。
- 没有证据支持继续操作时，可以停止并说明不确定性；如果认为无法可靠配准，可以明确放弃，不必为了完成任务而猜测。

结束时
- 调用 finish 一次，并用最后一条消息给出：你是否认为配准完成；主要依据和剩余不确定性；如果放弃，说明原因。
- 最终配准结果以工作台中的实际状态为准，不要只在文字里描述一个没有应用的变换。
'''


def write_prompts(cases, sandbox):
    pdir = STUDY / 'prompts'
    pdir.mkdir(exist_ok=True)
    cli = str(sandbox / 'workbench_cli.py')
    for c in cases:
        txt = prompt(c['case'], c['scene'], c['tag'], c['ref'], c['target'], cli)
        (pdir / (c['case'] + '.txt')).write_text(txt, encoding='utf-8')
        (sandbox / 'sessions' / c['case'] / 'TASK.md').write_text(txt, encoding='utf-8')
    print('wrote', len(cases), 'prompts to', pdir, 'and TASK.md per session')


if __name__ == '__main__':
    sb = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SANDBOX
    cs = build(sb)
    write_prompts(cs, sb)
