# MV-Recon interactive-registration workbench

A reproducible package for the 40-instance study “can a language model operating an
interactive 3D workbench beat a strong traditional registration pipeline on mono-depth
point clouds?”

**Result (all 40 instances, 19 unique view pairs):** the agent reached 14/40 joint
successes versus 24/40 for SIFT-rigid + reprojection refinement. Median error was
competitive (2.36°/0.26 m vs 1.93°/0.29 m), and the agent produced no success flips that the
baseline did not also achieve. Full report: `results/SUMMARY.md`.

## What is in here

```
data/<scene>/        agent-visible inputs only: RGB, each view's own predicted depth, K
sessions/<case>/     TASK.md, the initial state (deliberate unknown displacement), empty trace
run.py               the workbench command line (one command per process, JSON output)
workbench/           frozen, original CLI plus the baseline and export scripts
results/             report, per-sample data, baselines, agent results, checks
```

`data/` and `sessions/` contain **no ground-truth poses**. The true camera transforms used
for scoring are not in this package.

## Try the workbench

Requirements: Python 3.10+, `numpy`, `scipy`, `opencv-python`, `open3d`, `matplotlib`.

```powershell
python run.py courtyard_v2_eth3d_0008 stats
```

```powershell
python run.py courtyard_v2_eth3d_0008 photos
```

```powershell
python run.py courtyard_v2_eth3d_0008 match --matcher sift
```

```powershell
python run.py courtyard_v2_eth3d_0008 solve rigid3d
```

Each command prints JSON; a JSON `images` field names a PNG you can open. Move the cloud
with `rotate`, `translate`, `icp`, `fpfh`, `solve`; restrict matches with `mask` and
`--ref-box/--mov-box`; revert with `undo`; end with
`finish --status done|abstain --reason "..."`. Run `python run.py <case> state` for the case
id, budget and current matrix. List the 40 case ids with `dir sessions` (one folder per case).

To let a model drive it (as in the study), give the model only:

> Read and follow `sessions/<case>/TASK.md`, calling `python run.py <case> <command>`.

## Reproduce the analysis

`workbench/baselines_full.py` recomputes the traditional baselines (it needs the ground-truth
transforms, which live in the study repository's `eval_data/`, not here).
`workbench/export_clouds.py` regenerates the 3D exports when pointed at the study directories.

## Exported point clouds

See the study's `export/` folder: `index.html` lists every case; each scene has an offline
3D viewer (Plotly embedded) and coloured PLY files (`*_agent.ply`, `*_ref.ply`,
`*_input.ply`). Red = moving cloud at the agent's final transform; photo-coloured = reference.

## Configuration

```json
{"budget": {"photo": 6, "render": 12, "closeup": 6, "match": 5, "mask": 4,
            "transform": 8, "solve": 6, "icp": 4, "fpfh": 2, "total": 30}}
```

Each session starts with this budget; `state`, `stats`, `undo` and `finish` are free.

## Licence and provenance

Data is ETH3D (CC BY-NC-SA 4.0); depth is a frozen monocular prediction from the study's
DA3-metric pipeline. The workbench code in `workbench/` is the exact file used for the
reported sessions (its SHA-256 is in `PACKAGE_MANIFEST.json`), with paths untouched; `run.py`
is the same code with its data/session roots moved to this package.
