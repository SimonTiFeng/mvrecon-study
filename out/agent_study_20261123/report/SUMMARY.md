# MV-Recon interactive registration study — full-set report

**Date:** 2026-11-23  
**Task set:** 40 target instances, 19 unique view pairs, nested v2/v3/v4  
**Agent model:** Claude Sonnet 5, one fresh session per case  
**Protocol:** initial injected displacement D; score final cumulative M against T_true; success = rotation < 2° and lateral translation < 0.5 m.


## 1. What was run

The interactive workbench exposed only agent-visible data (two RGB views, each view’s own predicted depth, intrinsics, current point-cloud state) and generic tools: region-selecting SIFT/ORB/AKAZE matching, translation / rigid-3D / small-rotation / reprojection solvers, point-to-plane ICP, FPFH+RANSAC+ICP, manual rotate/translate, masking, undo and abstain. Budget 30 charged operations per case.

Each case was a separate subagent with a fresh context. The harness was audited: the subagent transcripts contain no project memory index, no parent user prompt, and no `eval_data` / `T_true` access. Ground truth was read only after the agent finished.


## 2. Headline comparison (all 40 instances)

| method | n | rot median ° | rot <2° | lat median m | lat <0.5 m | success |
|---|---|---|---|---|---|---|
| A_fpfh_icp | 40 | 4.57 | 14/40 | 2.93 | 9/40 | 6/40 |
| B_sift_rigid | 40 | 2.14 | 18/40 | 0.33 | 22/40 | 18/40 |
| C_sift_rigid_sky | 40 | 1.95 | 21/40 | 0.33 | 24/40 | 21/40 |
| D_rotA_sift_t | 40 | 4.57 | 14/40 | 0.43 | 22/40 | 12/40 |
| E_rotA_sift_t_sky | 40 | 4.57 | 14/40 | 0.43 | 22/40 | 12/40 |
| F_rotA_sift_corr | 40 | 2.12 | 16/40 | 0.39 | 23/40 | 14/40 |
| G_rotA_sift_reproj | 40 | 1.93 | 24/40 | 0.29 | 27/40 | 24/40 |
| H_sift_rigid_reproj | 40 | 1.93 | 24/40 | 0.29 | 27/40 | 24/40 |
| identity | 40 | 57.95 | 0/40 | 1.93 | 5/40 | 0/40 |
| AGENT | 40 | 2.36 | 16/40 | 0.26 | 29/40 | 14/40 |

No method was run only on a GT-selected subset. The coarse-rotation “<20°” view is reported separately in §4 using the explicit rule “A_fpfh_icp rotation <20°”.


## 3. Per-scene and per-subset


**A_fpfh_icp**

| scene | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| courtyard | 5 | 1.49 | 3/5 | 3.31 | 0/5 | 0/5 |
| delivery_area | 5 | 1.48 | 3/5 | 0.21 | 3/5 | 3/5 |
| electro | 6 | 52.69 | 3/6 | 4.17 | 0/6 | 0/6 |
| facade | 6 | 6.34 | 2/6 | 3.54 | 1/6 | 0/6 |
| office | 6 | 84.16 | 1/6 | 1.92 | 1/6 | 1/6 |
| playground | 6 | 16.59 | 0/6 | 3.11 | 0/6 | 0/6 |
| terrace | 6 | 2.07 | 2/6 | 0.27 | 4/6 | 2/6 |

| tag | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| v2 | 7 | 1.28 | 5/7 | 2.33 | 3/7 | 3/7 |
| v3 | 14 | 3.13 | 4/14 | 1.81 | 3/14 | 1/14 |
| v4 | 19 | 9.24 | 5/19 | 3.13 | 3/19 | 2/19 |

**B_sift_rigid**

| scene | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| courtyard | 5 | 1.97 | 5/5 | 0.33 | 5/5 | 5/5 |
| delivery_area | 5 | 5.59 | 2/5 | 1.26 | 2/5 | 2/5 |
| electro | 6 | 49.65 | 0/6 | 0.75 | 3/6 | 0/6 |
| facade | 6 | 10.08 | 0/6 | 2.05 | 0/6 | 0/6 |
| office | 6 | 0.86 | 5/6 | 0.04 | 6/6 | 5/6 |
| playground | 6 | 14.15 | 3/6 | 0.46 | 3/6 | 3/6 |
| terrace | 6 | 3.75 | 3/6 | 0.33 | 3/6 | 3/6 |

| tag | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| v2 | 7 | 1.97 | 4/7 | 0.24 | 5/7 | 4/7 |
| v3 | 14 | 2.06 | 7/14 | 0.30 | 8/14 | 7/14 |
| v4 | 19 | 5.59 | 7/19 | 0.62 | 9/19 | 7/19 |

**C_sift_rigid_sky**

| scene | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| courtyard | 5 | 1.95 | 5/5 | 0.33 | 5/5 | 5/5 |
| delivery_area | 5 | 3.72 | 2/5 | 0.85 | 2/5 | 2/5 |
| electro | 6 | 49.49 | 3/6 | 0.72 | 3/6 | 3/6 |
| facade | 6 | 8.68 | 0/6 | 1.77 | 0/6 | 0/6 |
| office | 6 | 0.86 | 5/6 | 0.04 | 6/6 | 5/6 |
| playground | 6 | 14.14 | 3/6 | 0.46 | 3/6 | 3/6 |
| terrace | 6 | 2.64 | 3/6 | 0.25 | 5/6 | 3/6 |

| tag | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| v2 | 7 | 1.82 | 5/7 | 0.18 | 5/7 | 5/7 |
| v3 | 14 | 1.95 | 8/14 | 0.29 | 9/14 | 8/14 |
| v4 | 19 | 3.72 | 8/19 | 0.47 | 10/19 | 8/19 |

**G_rotA_sift_reproj**

| scene | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| courtyard | 5 | 1.89 | 3/5 | 0.29 | 3/5 | 3/5 |
| delivery_area | 5 | 1.40 | 5/5 | 0.23 | 5/5 | 5/5 |
| electro | 6 | 56.52 | 0/6 | 1.65 | 3/6 | 0/6 |
| facade | 6 | 3.85 | 2/6 | 0.71 | 2/6 | 2/6 |
| office | 6 | 0.49 | 6/6 | 0.01 | 6/6 | 6/6 |
| playground | 6 | 15.54 | 3/6 | 0.83 | 3/6 | 3/6 |
| terrace | 6 | 1.98 | 5/6 | 0.15 | 5/6 | 5/6 |

| tag | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| v2 | 7 | 1.89 | 5/7 | 0.23 | 6/7 | 5/7 |
| v3 | 14 | 1.92 | 9/14 | 0.26 | 10/14 | 9/14 |
| v4 | 19 | 1.98 | 10/19 | 0.29 | 11/19 | 10/19 |

**H_sift_rigid_reproj**

| scene | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| courtyard | 5 | 1.89 | 3/5 | 0.29 | 3/5 | 3/5 |
| delivery_area | 5 | 1.40 | 5/5 | 0.23 | 5/5 | 5/5 |
| electro | 6 | 50.00 | 0/6 | 0.77 | 3/6 | 0/6 |
| facade | 6 | 3.85 | 2/6 | 0.71 | 2/6 | 2/6 |
| office | 6 | 0.49 | 6/6 | 0.01 | 6/6 | 6/6 |
| playground | 6 | 13.44 | 3/6 | 0.46 | 3/6 | 3/6 |
| terrace | 6 | 1.98 | 5/6 | 0.15 | 5/6 | 5/6 |

| tag | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| v2 | 7 | 1.89 | 5/7 | 0.23 | 6/7 | 5/7 |
| v3 | 14 | 1.92 | 9/14 | 0.26 | 10/14 | 9/14 |
| v4 | 19 | 1.98 | 10/19 | 0.29 | 11/19 | 10/19 |

**AGENT**

| scene | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| courtyard | 5 | 2.36 | 0/5 | 0.29 | 5/5 | 0/5 |
| delivery_area | 5 | 1.47 | 4/5 | 0.26 | 5/5 | 4/5 |
| electro | 6 | 47.15 | 0/6 | 0.81 | 3/6 | 0/6 |
| facade | 6 | 3.21 | 2/6 | 0.64 | 2/6 | 0/6 |
| office | 6 | 2.55 | 3/6 | 0.08 | 6/6 | 3/6 |
| playground | 6 | 22.26 | 3/6 | 0.52 | 3/6 | 3/6 |
| terrace | 6 | 1.20 | 4/6 | 0.18 | 5/6 | 4/6 |

| tag | n | rot med | <2° | lat med | <0.5 m | success |
|---|---|---|---|---|---|---|
| v2 | 7 | 1.47 | 4/7 | 0.20 | 6/7 | 4/7 |
| v3 | 14 | 2.10 | 6/14 | 0.25 | 11/14 | 5/14 |
| v4 | 19 | 3.22 | 6/19 | 0.29 | 12/19 | 5/19 |

## 4. Unique pairs and the GT-conditioned subset

The 40 instances repeat 19 scene/view pairs; D differs between nested instances, so their results are not copies. Deduplicated by taking the lowest subset tag per pair:

| method | n | rot median | <2° | lat median | <0.5 m | success |
|---|---|---|---|---|---|---|
| A_fpfh_icp | 19 | 5.69 | 6/19 | 2.35 | 4/19 | 3/19 |
| B_sift_rigid | 19 | 5.59 | 7/19 | 0.62 | 9/19 | 7/19 |
| C_sift_rigid_sky | 19 | 3.72 | 8/19 | 0.47 | 10/19 | 8/19 |
| D_rotA_sift_t | 19 | 5.69 | 6/19 | 0.59 | 9/19 | 5/19 |
| E_rotA_sift_t_sky | 19 | 5.69 | 6/19 | 0.58 | 9/19 | 5/19 |
| F_rotA_sift_corr | 19 | 3.32 | 7/19 | 0.42 | 10/19 | 6/19 |
| G_rotA_sift_reproj | 19 | 1.98 | 10/19 | 0.29 | 11/19 | 10/19 |
| H_sift_rigid_reproj | 19 | 1.98 | 10/19 | 0.29 | 11/19 | 10/19 |
| identity | 19 | 59.69 | 0/19 | 2.44 | 2/19 | 0/19 |
| AGENT | 19 | 2.91 | 6/19 | 0.29 | 11/19 | 5/19 |

**Reporting subset:** A_fpfh_icp rotation error < 20 deg (GT-based; reporting only) (n=26). This is a *reporting* split, not a run filter.

| method | n | rot median | <2° | lat median | <0.5 m | success |
|---|---|---|---|---|---|---|
| A_fpfh_icp | 26 | 1.65 | 14/26 | 1.81 | 9/26 | 6/26 |
| B_sift_rigid | 26 | 2.06 | 13/26 | 0.30 | 16/26 | 13/26 |
| C_sift_rigid_sky | 26 | 1.95 | 16/26 | 0.29 | 18/26 | 16/26 |
| D_rotA_sift_t | 26 | 1.65 | 14/26 | 0.28 | 21/26 | 12/26 |
| E_rotA_sift_t_sky | 26 | 1.65 | 14/26 | 0.27 | 21/26 | 12/26 |
| F_rotA_sift_corr | 26 | 1.82 | 16/26 | 0.29 | 21/26 | 14/26 |
| G_rotA_sift_reproj | 26 | 1.92 | 18/26 | 0.26 | 21/26 | 18/26 |
| H_sift_rigid_reproj | 26 | 1.92 | 18/26 | 0.26 | 21/26 | 18/26 |
| identity | 26 | 50.63 | 0/26 | 2.26 | 2/26 | 0/26 |
| AGENT | 26 | 1.82 | 13/26 | 0.26 | 23/26 | 11/26 |

## 5. Agent behaviour

- Instances finished: 38; abstained: 2; no finish: 0.
- Trajectory reconstruction passed: 40/40 (max replay error 3.55e-15).
- Median charged operations: 16.0 of 30; median trace actions: 19.0.
- Totals: match 122, solve 85, ICP 81, FPFH 14, undo 46.
- State-change sequences: {"rigid3d_then_icp": 37, "fpfh_start": 2, "other": 1}.
- Final transform identical (≤0.1°, ≤1 cm) to a traditional baseline in 3/40 cases; closest baseline counts: {"C_sift_rigid_sky": 3, "B_sift_rigid": 11, "E_rotA_sift_t_sky": 9, "F_rotA_sift_corr": 4, "H_sift_rigid_reproj": 5, "D_rotA_sift_t": 6, "G_rotA_sift_reproj": 2}.

## 6. Paired tests vs each baseline

Wilcoxon signed-rank on the 40 paired errors; success by exact McNemar.

| baseline | rot Δ median | p | lat Δ median | p | success A/B | McNemar p |
|---|---|---|---|---|---|---|
| A_fpfh_icp | -0.269 | 0.0152 | -1.657 | 3.01e-05 | 14/6 | 0.00781 |
| B_sift_rigid | -0.553 | 0.0148 | -0.002 | 0.531 | 14/18 | 0.388 |
| C_sift_rigid_sky | -0.537 | 0.0301 | -0.002 | 0.679 | 14/21 | 0.118 |
| D_rotA_sift_t | -0.269 | 0.0152 | -0.131 | 0.0436 | 14/12 | 0.791 |
| E_rotA_sift_t_sky | -0.269 | 0.0152 | -0.130 | 0.0436 | 14/12 | 0.791 |
| F_rotA_sift_corr | -0.008 | 0.154 | -0.060 | 0.0344 | 14/14 | 1 |
| G_rotA_sift_reproj | -0.021 | 0.685 | -0.008 | 0.984 | 14/24 | 0.00195 |
| H_sift_rigid_reproj | -0.110 | 0.289 | +0.000 | 0.738 | 14/24 | 0.00195 |
| identity | -48.355 | 9.78e-08 | -1.435 | 0.000229 | 14/0 | 0.000122 |

## 7. Cases where the agent differs from the strongest baseline

Δ is agent minus H_sift_rigid_reproj (negative is better).

| case | status | agent rot / lat | H rot / lat | Δlat | sequence |
|---|---|---|---|---|---|
| facade_v4_eth3d_0201 | done | 7.66° / 1.49 m | 73.93° / 31.72 m | -30.23 | solve:rigid3d, solve:reproj, fpfh, icp, solve:translation, solve:reproj, fpfh, icp, translate |
| playground_v4_eth3d_0249 | done | 56.62° / 0.96 m | 88.53° / 2.44 m | -1.48 | solve:rigid3d, fpfh, icp |
| terrace_v4_eth3d_0228 | done | 89.87° / 2.81 m | 175.93° / 4.27 m | -1.46 | fpfh, translate, translate, rotate |
| facade_v3_eth3d_0209 | done | 3.44° / 0.18 m | 3.85° / 0.71 m | -0.53 | solve:rigid3d, icp, solve:rigid3d, icp, fpfh |
| facade_v4_eth3d_0209 | done | 3.44° / 0.19 m | 3.85° / 0.71 m | -0.52 | solve:rigid3d, icp, solve:rigid3d, solve:reproj, solve:rigid3d, icp, solve:reproj, icp, solve:reproj, fpfh |
| courtyard_v4_eth3d_0000 | done | 2.91° / 0.41 m | 5.44° / 0.57 m | -0.16 | solve:rigid3d, icp |
| courtyard_v3_eth3d_0000 | done | 2.91° / 0.41 m | 5.44° / 0.57 m | -0.16 | solve:rigid3d, icp, icp |
| facade_v2_eth3d_0209 | done | 2.98° / 0.56 m | 3.85° / 0.71 m | -0.15 | solve:rigid3d, icp, solve:reproj, solve:rigid3d, solve:reproj, icp, solve:reproj |
| terrace_v2_eth3d_0236 | done | 0.89° / 0.08 m | 1.98° / 0.15 m | -0.07 | solve:rigid3d, icp, solve:reproj, icp |
| terrace_v3_eth3d_0236 | done | 0.89° / 0.08 m | 1.98° / 0.15 m | -0.07 | solve:rigid3d, icp, icp |
| terrace_v4_eth3d_0236 | done | 0.41° / 0.09 m | 1.98° / 0.15 m | -0.06 | solve:rigid3d, icp, icp |
| electro_v3_eth3d_0097 | done | 2.14° / 0.24 m | 2.84° / 0.29 m | -0.05 | solve:rigid3d, icp, solve:reproj, icp, icp |
| delivery_area_v2_eth3d_0081 | done | 1.47° / 0.20 m | 1.40° / 0.23 m | -0.03 | solve:rigid3d, icp, solve:reproj, icp |
| delivery_area_v3_eth3d_0081 | done | 1.47° / 0.20 m | 1.40° / 0.23 m | -0.03 | solve:rigid3d, icp, icp, solve:reproj |
| electro_v4_eth3d_0097 | done | 2.73° / 0.26 m | 2.84° / 0.29 m | -0.03 | solve:rigid3d, icp, solve:reproj, icp |
| electro_v2_eth3d_0097 | done | 2.73° / 0.26 m | 2.84° / 0.29 m | -0.03 | solve:rigid3d, icp, solve:rigid3d, icp |
| playground_v4_eth3d_0286 | done | 1.10° / 0.07 m | 0.28° / 0.09 m | -0.02 | solve:rigid3d, icp, solve:reproj, icp, icp |
| playground_v2_eth3d_0286 | done | 1.03° / 0.07 m | 0.28° / 0.09 m | -0.02 | solve:rigid3d, icp, solve:reproj, icp, solve:rigid3d, fpfh |
| playground_v3_eth3d_0286 | done | 0.92° / 0.07 m | 0.28° / 0.09 m | -0.02 | solve:translation, icp, icp, icp, solve:reproj, solve:translation, icp, solve:rotcorr |
| office_v4_eth3d_0133 | done | 1.12° / 0.03 m | 0.23° / 0.01 m | +0.03 | solve:rigid3d, icp, solve:reproj, icp |
| office_v3_eth3d_0133 | done | 0.70° / 0.04 m | 0.23° / 0.01 m | +0.03 | solve:rigid3d, icp, solve:reproj |
| delivery_area_v4_eth3d_0081 | done | 1.58° / 0.27 m | 1.40° / 0.23 m | +0.04 | solve:rigid3d, icp, icp, solve:reproj |
| office_v2_eth3d_0133 | done | 0.52° / 0.05 m | 0.23° / 0.01 m | +0.04 | solve:rigid3d, icp |
| office_v3_eth3d_0149 | done | 6.51° / 0.11 m | 0.75° / 0.02 m | +0.09 | solve:rigid3d, icp |
| office_v4_eth3d_0149 | done | 3.98° / 0.13 m | 0.75° / 0.02 m | +0.11 | solve:rigid3d, icp, solve:reproj, icp |
| terrace_v3_eth3d_0231 | done | 2.06° / 0.27 m | 1.93° / 0.15 m | +0.12 | solve:rigid3d, icp, icp, solve:rigid3d, solve:rigid3d, icp, solve:reproj |
| delivery_area_v3_eth3d_0057 | done | 0.69° / 0.26 m | 1.03° / 0.12 m | +0.14 | solve:rigid3d, icp, solve:reproj, icp, fpfh, solve:rotcorr, solve:translation, solve:rotcorr |
| terrace_v4_eth3d_0231 | done | 1.52° / 0.29 m | 1.93° / 0.15 m | +0.14 | solve:rigid3d, icp, icp |
| office_v4_eth3d_0152 | done | 4.94° / 0.15 m | 1.11° / 0.02 m | +0.14 | solve:rigid3d, icp, icp, solve:reproj, icp |
| delivery_area_v4_eth3d_0057 | done | 3.22° / 0.31 m | 1.03° / 0.12 m | +0.19 | solve:rigid3d, icp, solve:reproj, solve:rigid3d, icp, icp |
| facade_v3_eth3d_0169 | done | 1.32° / 0.72 m | 1.92° / 0.36 m | +0.35 | solve:rigid3d, icp |
| facade_v4_eth3d_0169 | done | 1.32° / 0.72 m | 1.92° / 0.36 m | +0.35 | solve:rigid3d, icp, icp |
| electro_v4_eth3d_0126 | done | 91.57° / 4.41 m | 100.46° / 1.26 m | +3.15 | solve:rigid3d, fpfh, icp, translate, translate, icp, icp |
| playground_v4_eth3d_0278 | abstain | 43.41° / 4.99 m | 63.78° / 1.57 m | +3.42 | solve:rigid3d, fpfh, solve:reproj, icp, fpfh, solve:reproj, icp |
| playground_v3_eth3d_0278 | done | 55.21° / 5.70 m | 26.61° / 0.84 m | +4.85 | solve:rigid3d, fpfh, solve:reproj, translate, icp, icp, solve:translation, icp, icp |
| electro_v3_eth3d_0126 | done | 91.99° / 10.11 m | 97.16° / 3.18 m | +6.92 | solve:rigid3d, fpfh, icp, solve:rigid3d, icp |

## 8. Direct answers

- **Which operations did the agent choose?** {"rigid3d_then_icp": 37, "fpfh_start": 2, "other": 1}. The dominant sequence was SIFT match → rigid-3D RANSAC → ICP; FPFH was used when SIFT failed, and reprojection was used, accepted, or undone case by case.
- **Where did it improve or degrade?** It crossed the success threshold on 0 cases where H failed; H crossed it on 10 cases where the agent failed. Success counts: agent 14, H 24 (of 40).
  - H-only successes: courtyard_v2_eth3d_0008, courtyard_v3_eth3d_0008, courtyard_v4_eth3d_0008, delivery_area_v4_eth3d_0057, facade_v3_eth3d_0169, facade_v4_eth3d_0169, office_v3_eth3d_0149, office_v4_eth3d_0149, office_v4_eth3d_0152, terrace_v3_eth3d_0231.
- **Was there a real gain over the strongest baseline?** On the full set, no: the agent is competitive on median lateral error but converts fewer cases into joint rotation+lateral successes, and it uses 5–20× the operations of the fixed pipeline. It did avoid H’s catastrophic failures on facade_v4_0201 (7.66°/1.49 m vs 73.93°/31.72 m) and improved both facade 0209 instances, but rotation stayed above the 2° threshold there, so these are error reductions, not success flips.
- **Did any gain come from multi-round judgment rather than the tools?** The numerical work is done by SIFT/FPFH/ICP/reprojection. Against its own first solver output, later rounds improved rotation in 18 and worsened it in 15 cases; improved lateral in 18 and worsened it in 14. The final transform is identical to a traditional baseline in 3 of 40 cases. The model’s measurable contribution is tool selection and rejecting bad refinements, not better numerics.
- **Unfinished items, leakage risk and limitations.** 2 abstention(s). The run had no GT in the agent context; the audit found no memory index, parent prompt, eval_data or T_true access in any transcript. One of 40 transcripts contains an inline numpy matrix-arithmetic command outside the workbench CLI (no GT, no external data); its final state is still the workbench state and is reported, with and without it unchanged in conclusion. Residual risks: the subagent’s cwd is the project directory (its name is visible, though no project file was read), one sample per case (no model-stochasticity average), and depth noise dominates several scenes. Interrupted agents were resumed from their own workbench state; cases with no finish are reported as no_finish, not dropped.

## 9. Evidence and limitations

- Raw: `raw/transcripts/`, `raw/final_reports/`, `raw/launches/`; workbench traces and state: the sandbox `sessions/`; scored data: `eval/`.
- The workbench enforces budgets, serialises parallel tool calls with a per-session lock, and logs every operation with before/after matrices; final transforms replay exactly.
- Two harness defects were found and fixed before the reported run: a lost-update race between parallel CLI calls, and a `closeup` JSON serialisation error. All reported sessions use one frozen CLI (hash in `study/sandbox_manifest.json`).
- The Agent-tool subagents run with cwd in the project directory; their system context contains the directory name but not the memory index or the parent prompt. Any Bash command other than the workbench CLI is flagged by `eval/leakage_audit.json`.
- The agent may decline; abstentions are scored on the final state and reported, not dropped.
- No repeated runs per case: model stochasticity is not averaged. This is a single sample of 40 independent sessions, not a confidence interval on the model.