# Corrected results

Scoring rule: score the CUMULATIVE matrix M against `T_true` (never against the
incremental A = T @ inv(D)).

Denominator 40 targets: per scene 1 (v2) + 2 (v3) + up to 3 (v4). courtyard and
delivery_area have no usable 4th view, so they contribute 5 instead of 6
(5+5+6+6+6+6+6 = 40).

| method | rot median | trans median | rot < 20 deg | trans < 0.5 m | both |
|---|---:|---:|---:|---:|---:|
| identity (do nothing) | 57.95 deg | 2.536 m | 0/40 | 0/40 | 0/40 |
| FPFH+RANSAC+ICP | **5.02 deg** | 2.670 m | **25/40** | 6/40 | **6/40** |
| scripted closed loop | 17.23 deg | 2.442 m | 22/40 | 4/40 | 4/40 |

## This overturns my earlier headline

Under the buggy scorer I reported "no method ever gets inside 20 degrees; traditional
registration fails completely." That was an artefact of the two bugs. With the correct
convention FPFH+RANSAC+ICP recovers rotation on 25/40 targets at a 5-degree median.

What stays hard is **translation**: 6/40 within 0.5 m. That is the real headroom, and it
is a different claim from the earlier one.

## Per scene (median rotation error, degrees)

| scene | identity | FPFH+ICP | FPFH+ICP rot<20 |
|---|---:|---:|---:|
| courtyard | 27.5 | 1.5 | 5/5 |
| delivery_area | 54.3 | 1.7 | 4/5 |
| facade | 54.8 | 3.4 | 5/6 |
| playground | 49.2 | 16.4 | 3/6 |
| terrace | 73.6 | 44.5 | 3/6 |
| electro | 79.8 | 62.7 | 3/6 |
| office | 66.0 | 95.8 | 1/6 |

office is the one scene where the classical baseline is worse than doing nothing on
rotation (95.8 vs 66.0 deg): a genuine registration failure.

## The scripted loop does not beat the baseline

17.23 vs 5.02 deg median rotation; 22/40 vs 25/40 inside 20 deg. The policy uses no image
semantics (same central uv box in both views, local nearest-neighbour alignment), which is
exactly the "fixed correspondences then ICP" pattern the spec warns against. It is a
harness test, not the semantic agent the study is about.

## Provenance

Every row of `runs/baseline_corrected.json` and `runs/agent_runs_corrected.json` stores the
initial matrix, the final cumulative matrix, `T_true`, `D_injected`, and for ICP the coarse
and fine transforms, plus the full agent observation/action trace.
`runs/correction_audit.json` reports zero mismatches against the current task.
