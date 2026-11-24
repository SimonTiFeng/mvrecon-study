# Publishing the MV-Recon study to GitHub

Ready-to-run PowerShell guide. Nothing here has been executed — `gh` is not installed on this
machine and no repository has been created. Run the commands yourself when you want to publish.

## What you have

| Bundle | Path (relative to `out/agent_study_20261123/`) | Size |
|---|---|---|
| 3D results (PLY + viewers) | `report/mvrecon-agent-3d-results.zip` | ~97 MB |
| Full workbench package | `package/mvrecon-workbench/` (→ `report/mvrecon-workbench.zip`) | ~3.0 GB (≈1.1 GB zipped) |

## Size limits — read first

- **GitHub repos**: files > 100 MB are rejected outright; keeping a repo under ~5 GB total and
  pushes under ~1 GB is what stays practical. The ~3 GB workbench package (≈1.1 GB zipped)
  **cannot** go into a normal repo, even with LFS on a free plan (1 GB LFS storage / 1 GB
  bandwidth per month).
- **Practical options** (pick one):
  1. **Recommended**: publish code + docs + the PLY results bundle in the repo (~97 MB, under the
     100 MB per-file limit — no LFS needed), and attach `mvrecon-workbench.zip` as a **GitHub
     Release asset** (the 1.1 GB zip is under GitHub's 2 GB per-asset limit).
  2. Publish only the code, scripts and report; host both zips on Zenodo or institutional storage
     and link them from the README.
  3. Paid LFS (e.g. GitHub Data Packs) if you really need the full package in git history.

## 1. Initialise the repository

```powershell
cd C:\Users\Simon\Downloads\Git\research\music\depth_calib\mvrecon
git init -b main
git lfs install
```

## 2. Choose what goes in git

Create `.gitignore` so the heavy generated trees never enter history:

```powershell
@'
out/**/package/
out/**/export/
out/**/*.zip
*.ply
*.npy
__pycache__/
'@ | Set-Content .gitignore -Encoding utf8
```

If you do want point clouds or arrays versioned, track them with LFS **before** the first `git add`:

```powershell
git lfs track "*.npy" "*.png" "*.ply"
git add .gitattributes
```

## 3. Commit and push

```powershell
git add .
git commit -m "MV-Recon interactive registration study: code, package builder, results report"
git remote add origin https://github.com/<your-account>/mvrecon-study.git
git push -u origin main
```

Create the empty repo on github.com first (no README, no .gitignore), or with `gh repo create
mvrecon-study --private --source .` once `gh` is installed.

## 4. Release asset for the big bundle (recommended for the workbench)

On github.com → your repo → **Releases** → *Draft a new release* → tag `v1.0` → drop in:

- `out/agent_study_20261123/report/mvrecon-workbench.zip` (≈1.1 GB)
- optionally `out/agent_study_20261123/report/mvrecon-agent-3d-results.zip`

Each asset must be under 2 GB. Then link them from the repo README:

```markdown
Full workbench (40 sessions, agent-visible data, frozen CLI): see Releases → v1.0
```

## 5. If you insist on the workbench inside git

Only with a paid LFS quota:

```powershell
git lfs track "out/agent_study_20261123/package/**"
git add .gitattributes
git add "out/agent_study_20261123/package/mvrecon-workbench"
git commit -m "workbench package via LFS"
git lfs push --all origin main
git push origin main
```

Expect a multi-GB LFS upload and quota burn; the Release-asset route is cheaper and simpler.

## Notes

- The workbench package contains **agent-visible data only** (RGB, own-view depth, K) — no
  ground-truth poses — so it is safe to share.
- The export bundle is fully offline (viewers are self-contained Plotly HTML, no network).
- A 97 MB results zip committed directly is fine; unzipped PLYs (120 files) via LFS are fine too,
  but the zip is simplest.
