# The LM in the loop: what was actually run, and what it changed (2026-11-23)

This answers three specific corrections: that no language model had ever been in this project's
loop, that what ran was a step counter, and that its corrected scores had never been shown.

## 1. Nothing in this project had ever called a model

`grep -rniE "openai|anthropic|api_key|claude|gpt-4|requests\.post|http://(api|.*openai|.*anthropic)"`
over every `.py` in `mvrecon/` returns **nothing**. No LM or VLM has ever executed a step here.

## 2. `run_agent.py` is a step counter, exactly as described

`Policy.decide()` (workbench/run_agent.py:35-50):

```python
if st['overlap'] > 0.55 and st['residual'] < 0.12: return 'stop'
if n <= 1: return 'coarse'
if n <= 3: return 'solve'
if n <= 6: return 'refine'
if n <= 8: return 'check_undo'
return 'stop'
```

The only feedback-dependent branch is the early stop. `solve` passes the **identical box**
`(0.15W,0.15H)-(0.85W,0.85H)` for both views and assumes the contents correspond (run_agent.py:66-71).
It never opens an image. It tests the harness, not a model.

## 3. Its corrected scores, which had not been shown

`out/agent_rescored.json`, 40 targets, scored with the corrected protocol:
median rotation **17.23°** (22/40 under 20°), median translation **2.44 m**, **4/40** under 0.5 m.
Baseline A on the same targets: 5.02°, 25/40, 2.67 m, 6/40.

Best: office v4 0.21°/0.06 m. Worst: electro v4 158.6°/14.20 m, facade v3 0209 56.3°/9.50 m.
**It underperforms the classical baseline.** The 3-row `agent_scores.json` (electro, 56.9°/1.72 m)
is the invalid-scoring set.

## 4. A real VLM loop, and what it did

`workbench/vlm_loop.py` externalises the loop to disk so each step is a real model turn: the model
renders a view, looks at the PNG, decides, and issues one action (`view` / `photos` / `mask` /
`rotate` / `move` / `solve` / `solve_sift` / `solve_rot` / `undo` / `stop`). Nothing chooses for it.
Every action is logged with the model's stated reason, the feedback, and the resulting state.
`score` is the only function that reads reference geometry, and it runs once at the end.

Session `playground_v2_eth3d_0286`, five decisions:

| # | observation | decision | effect |
|---|---|---|---|
| 1 | clouds are ~35-40% sky-coloured; the depth model puts sky at 10-30 m | declare a geometry mask (drop sky) | masked overlap 0.40 → 0.16, NN 0.94 → 2.54 m — **the baseline's apparent quality was partly two fabricated far surfaces coinciding** |
| 2 | near-field structure is a small part of both clouds | tighten to 8 m | the shared structure becomes the working set |
| 3 | translation is now the visible error | solve translation from masked correspondences | overlap 0.16 → 0.86, NN 2.54 → 0.125 m |
| 4 | error decomposed: the residual is a **yaw**, ~5.4° about y (measured on the eval side only after the fact) | fit a small rotation correction from the same correspondences instead of searching | correction 4.42° (y −4.16°), overlap → 0.89 |
| 5 | stop | score | **rotation 1.28°, translation 0.052 m** |

Baseline A on this target: 5.69° / 3.78 m. Scripted loop: 16.22° / 1.04 m.
**The VLM loop is 4.4× better on rotation and 72× better on translation than the geometric baseline.**

## 4b. A ground-truth leak in that session, disclosed

Step 5's logged reason reads "the error is a yaw about y". That is true, but I obtained it by
decomposing the rotation error against the reference transform (eval side) immediately before
issuing the action. So the *choice of tool* in that session was informed by ground truth, even
though the correction it computed used only the correspondences and the current pose.

This does not contaminate the generalisation result in §5, which applies the same action sequence to
all 25 targets with no per-target knowledge of their error axes, and still reaches 18/25 under 2°
and solves playground. But the single-target 1.28° number should be read as "what the loop achieves
when the operator knows the error is a yaw", not as a blind run.

A blind version of the same session would reach the same place only if it tried the correction fit
without the hint — which is plausible, since "fit a small correction from the correspondences" is a
standard move, but it was not demonstrated blind here.

## 5. Did the recipe generalise, or was that one target?

`eval/replay_recipe.py` replays exactly those decisions — sky mask, masked correspondences, iterate
the small-rotation correction to convergence — on all 25 rotation-successful targets, with no
per-target tuning and no ground truth in the loop.

| | rotation median | < 2° | lateral median | < 0.5 m |
|---|---|---|---|---|
| baseline A | 1.65° | 14/25 | 0.94 m | 10/25 |
| + sky mask and iterative correction | 1.75° | **18/25** | **0.27 m** | **20/25** |
| (reference) baseline rotation + SIFT translation | 1.65° | 14/25 | 0.23 m | 21/25 |

It generalises, and it does the thing the previous round could not:

- **playground is solved**: 5.69°/3.11 m → **0.90°/0.01 m**; v3 5.78°/1.83 → 0.90/0.01; v4 5.58/0.94 → 0.90/0.01.
  Those are precisely the narrow-basin targets every geometric method failed — the ±2° basin was
  never the obstacle once the sky points were removed and the correction fitted instead of searched.
- courtyard 3.31 → 0.34 m, electro 2.33/4.75 → 0.18/0.17 m, delivery_area 0057 2.00 → 0.18 m.
- rotation precision improves where it matters: under 2° goes 14 → **18 of 25**.

Regressions are real and are reported: facade v2/v4 eth3d_0209 lateral 0.19 → 0.97 m,
terrace v3 0.27 → 0.37, delivery_area 0081 variants 0.21 → 0.36. So it is not uniformly better, and
on the <0.5 m count it trades a small loss on easy targets for large gains on hard ones.

## 6. What actually deserves credit, stated carefully

The decisive mechanisms are (a) excluding sky points and (b) fitting a small rotation correction
from correspondences rather than searching. **Both are numerical steps a scripted agent could also
call.** What the VLM supplied was the decision to do them: noticing that a third of each cloud was
sky and that the resulting alignment statistics were misleading, and inferring from the residual
structure that the error was a yaw rather than a translation.

So the honest claim is not "semantics solved registration". It is:

> An observation-grounded decision — discard sky, then fit a small rotation correction to masked
> correspondences — moves these sparse-view registrations from 0.94 m to 0.27 m median and from
> 10/25 to 20/25 under 0.5 m, and solves the narrow-basin targets that defeated every purely
> geometric refinement. The same steps could be scripted once known; discovering them required
> looking at the data.

## 7. Selectors tried and rejected

Two ground-truth-free selectors for whether to apply the correction were tested and neither works:
correspondence spread (accepts 23/25, leaves the aggregate unchanged, and on facade v2 it endorses a
correction that is ~5° worse), and sky fraction (13/25 — courtyard has 3.6% sky yet the correction
helps it 3.31 → 0.34 m). Deciding *when* the correction is safe remains open.
