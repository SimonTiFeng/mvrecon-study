"""Pareto frontier: AbsRel vs what each configuration needs at test time."""
import numpy as np
print("=== Configuration Pareto table ===")
print()
rows = [
 ("DA3 raw, no calibration",        0.1007, "0", "none"),
 ("+ per-image landmark",           0.0619, "0", "landmark cloud"),
 ("+ landmark & fixed 0.952",       0.0384, "0", "landmark cloud"),
 ("+ landmark & fitted constant",   0.0381, "1", "landmark cloud + other scenes' GT"),
 ("per-image oracle (unreachable)", 0.0353, "inf", "GT depth"),
]
print(f"{'configuration':<34}{'ETH3D':>8}{'deps trained':>14}{'needs at test':>30}")
for n,a,t,d in rows:
    print(f"{n:<34}{a:>8.4f}{t:>14}{d:>30}")
print()
print("=== KITTI ===")
rows2=[("DA3 raw",0.0986,"landmark cloud"),("+ LiDAR",0.0668,"landmark cloud")]
for n,a,d in rows2: print(f"  {n:<20}{a:>8.4f}   needs {d}")
print()
print("=== Which configurations are Pareto-optimal? ===")
print("Cost axis: number of things you must obtain at test time.")
print("  A: nothing (just run the model)          -> 0.1007")
print("  B: one landmark/LiDAR cloud per image    -> 0.0619")
print("  C: same + a fixed reference constant     -> 0.0384")
print("  D: same + GT from other scenes           -> 0.0381")
print()
print("C dominates B (better, same cost). C ties D (0.0384 vs 0.0381) at strictly")
print("lower cost, so D is NOT Pareto-optimal: it buys 0.0003 for a GT dependency.")
print("The frontier is {A, C}: A is cheapest, C is most accurate without GT.")
print()
print("Headline claim should therefore use C: 0.0384 on ETH3D, fully label-free.")
