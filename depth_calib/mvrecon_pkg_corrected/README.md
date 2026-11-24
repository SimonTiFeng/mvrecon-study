# MV-Recon corrected handoff v2

This package supersedes the v1 handoff. v1 had two independent defects: (1) the scorer
compared the cumulative transform M to incremental A=T@inv(D), not to T; (2) COLMAP
world-to-camera poses were converted with the wrong relative-pose formula. The original
v1 package and logs are preserved under audit/legacy_pre_fix. See audit/COORDINATE_AUDIT.md.

## Contents
- agent_input/: full-res RGB, K, raw depth cache and the actual own-camera-frame clouds.
  **No poses, D, T, or GT are in this tree.**
- evaluation_only/: corrected T_true, injected D, COLMAP poses, and PLYs for the actual
  start state and GT-aligned state. Do not expose this tree to an agent.
- runs/: corrected identity/FPFH+ICP/scripted-loop results. Each row includes initial M,
  final cumulative M, T, D, the coarse/fine ICP matrices, and complete agent hist/trace.
- code/: corrected generation, scoring, workbench, and regression-test code.
- audit/legacy_pre_fix/: exact v1 artefacts, retained for reproducibility, not valid scores.

## Correct transform convention
COLMAP supplies world->camera: p_cam=R p_world+t. Therefore camera j -> camera ref i is
R_ij=R_i R_j^T, t_ij=t_i-R_i R_j^T t_j.

The harness starts each cloud at M0=D. An incremental recovery A=T@inv(D) is left-composed,
so M_final=A@D=T. **Score the cumulative M_final against T_true.**

## Validation
`python code/bench/test_transform_convention.py` tests the formula and exact composition.
`runs/correction_audit.json` confirms all 80 baseline and 40 agent rows match the current
task matrices with zero mismatches.
