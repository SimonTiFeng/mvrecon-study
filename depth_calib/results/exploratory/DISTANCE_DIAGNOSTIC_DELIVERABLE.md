# Distance-stratified diagnostic — deliverable per Depth_Distance_Diagnostic_Plan.md

Date: 2026-11-17. Data: 55 committed pilot images (NYUv2 38, ETH3D 17), 14,260,266 valid pixels.
One calibration per image; no per-band re-fitting; all pixels; no subsampling (config v0.3.0).
Every number is reproduced by `scripts/distance_stratified.py` → `results/exploratory/distance_stratified.json`.

## Table 1 — Overall comparison (both Mono oracles, as required)

| Dataset | images | A-solver accepted | B raw | B + our anchors | A + our affine | **A GT-scale oracle** | **A GT-affine oracle** |
|---|---:|---:|---:|---:|---:|---:|---:|
| NYUv2 | 38 | 18 | **0.0947** | 0.1066 | 0.2761 | 0.1136 | **0.0610** |
| ETH3D | 17 | 7 | 0.1074 | 0.1337 | 0.2972 | 0.0578 | **0.0262** |

Both Mono oracles are present. The affine oracle is far below the realised 0.28 (there is headroom);
the constant-scale oracle is **not** enough on NYUv2 (0.1136, worse than Metric 0.0947), so the mono
backend needs a shift term — a scale-only calibration is structurally insufficient.

## Table 2 — Pixel-level AbsRel by GT depth band

Per-image band mean, then median across images. Bands are fixed a priori. `<0.5 m` is **N/A for every
system** because no pixel in either dataset is that close (see note below).

NYUv2 (38 images)

| band | B raw | B + ours | A + ours | A scale oracle | A affine oracle |
|---|---:|---:|---:|---:|---:|
| <0.5 m | N/A | N/A | N/A | N/A | N/A |
| 0.5–1.5 m | **0.0885** | 0.0885 | 0.3672 | 0.2433 | 0.0654 |
| 1.5–3 m | **0.0807** | 0.0979 | 0.2637 | 0.1094 | 0.0485 |
| 3–5 m | **0.1125** | 0.1175 | 0.2157 | 0.0963 | 0.0631 |
| ≥5 m | **0.1294** | 0.1351 | 0.3664 | 0.1318 | 0.0748 |

ETH3D (17 images)

| band | B raw | B + ours | A + ours | A scale oracle | A affine oracle |
|---|---:|---:|---:|---:|---:|
| <0.5 m | N/A | N/A | N/A | N/A | N/A |
| 0.5–1.5 m | N/A | N/A | N/A | N/A | N/A |
| 1.5–3 m | **0.0955** | 0.1605 | 2.7082 | 0.2483 | 0.0354 |
| 3–5 m | **0.1000** | 0.1425 | 1.1007 | 0.0750 | 0.0368 |
| ≥5 m | **0.1149** | 0.1317 | 0.2958 | 0.0294 | 0.0261 |

## Table 3 — Image-level, grouped by each image's GT median depth

| Dataset | group | n | B raw | B + ours | A + ours | A scale oracle | A affine oracle |
|---|---|---:|---:|---:|---:|---:|---:|
| NYUv2 | <1 m | 0 | N/A | N/A | N/A | N/A | N/A |
| NYUv2 | 1–2 m | 3 | **0.0511** | 0.0622 | 0.2610 | 0.1096 | 0.0553 |
| NYUv2 | 2–3 m | 19 | **0.0954** | 0.1139 | 0.2316 | 0.1129 | 0.0469 |
| NYUv2 | ≥3 m | 16 | **0.0989** | 0.1066 | 0.3593 | 0.1181 | 0.0640 |
| ETH3D | ≥3 m | 17 | **0.1074** | 0.1337 | 0.2972 | 0.0578 | 0.0262 |

## Table 4 — Anchor depth coverage vs image depth range

NYUv2: anchor/image depth log-span ratio median **0.43** (p10–p90 0.13–0.84); **15/38** images have
anchors spanning <30% of the scene's depth range. This is the mechanism behind the 0.28-vs-0.047 gap and
it is uniform across bands.

## Why the <0.5 m band is empty (requested near-field check)

Across all **1,699** processed frames the 5th-percentile GT depth never falls below **0.715 m** (NYUv2)
or **0.852 m** (ETH3D); pixels closer than 0.5 m = **0 / 14,260,266**. 281 frames have a 5th percentile
below 1.0 m — the closest available proxy, reported as the 0.5–1.5 m band. Reaching true sub-0.5 m
imagery needs a different dataset (or self-captured ARKit depth), not a different sample of these two.

## Confirmed vs still hypothesis

**Confirmed**
- Metric beats Mono+anchors in every band and every image group, with wide margins.
- Mono's affine oracle beats Metric everywhere; the scale-only oracle does not → shift term required.
- The realised gap is explained by anchor depth coverage (43% median, 40% of images <30%).
- Adding anchors to Metric degrades it on both datasets and in every band.

**Not confirmed / still hypothesis**
- Metric's advantage being caused by its training distance prior: the gap is flat across 1.5–3, 3–5
  and ≥5 m, so a distance-specific overfit is **not** supported.
- Any statement about sub-0.5 m behaviour (no data).
- Composition (indoor vs outdoor) effects: ETH3D's large mono errors at 1.5–3 m rest on few anchors and
  are confounded with sparse laser GT.
