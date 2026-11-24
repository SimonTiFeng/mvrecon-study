"""Workbench smoke test: load, transform, undo, restore, pick, stats."""
import sys, pathlib, numpy as np
sys.path.insert(0,'mvrecon/workbench')
from wb import Workbench

wb = Workbench('mvrecon/agent_data/office', stride=24)
print('views:', sorted(wb.clouds), 'ref:', wb.ref)
c0 = wb.clouds[sorted(wb.clouds)[0]]
print(f'  {c0.sid}: {len(c0.points)} pts, K fx={c0.K[0,0]:.1f}')
c1sid = [s for s in sorted(wb.clouds) if s != wb.ref][0]
c1 = wb.clouds[c1sid]
print(f'  target {c1sid}: {len(c1.points)} pts')

before = c1.transformed().copy()
theta=np.deg2rad(20)
R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
wb.apply(c1sid, R=R, t=np.array([0.5,0.2,-0.1]), s=1.3)
after = c1.transformed()
print(f'  after apply: moved {np.abs(after-before).max():.2f} m, scale {c1.s:.2f}')
print(f'  overlap {wb.overlap_stats(c1sid)}, residual {wb.residual(c1sid):.3f}')

ok = wb.undo()
restored = c1.transformed()
print(f'  undo -> {ok}, restored to baseline: {np.abs(restored-before).max():.2e}')
print(f'  state after undo: s={c1.s:.3f}, |t|={np.linalg.norm(c1.t):.3e}')

pick = wb.pick_region(wb.ref, (60,40,160,140))
print(f'  pick on {wb.ref}: {pick["n"]} real points from a 100x100 uv box')

# no-op transform on the reference must be refused
try:
    wb.apply(wb.ref, t=np.array([1,0,0]))
    print('  ERROR: reference moved')
except ValueError as e:
    print(f'  reference protection OK: {e}')
