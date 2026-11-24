# Export bundle

Two archives:

- `mvrecon-agent-3d-results.zip` — every agent's aligned point cloud, the reference cloud and the
  injected start state (PLY, photo-coloured, full 32-px sampling), one offline 3D viewer per scene
  and an `index.html`. Open `index.html` in any browser; no server or network needed.
  Red = moving cloud at the agent's final transform, photo-coloured = reference.
- `mvrecon-workbench.zip` — the self-contained workbench: agent-visible data (RGB, own-view depth,
  K), 40 initial session states with `TASK.md`, `run.py`, the frozen CLI, and the results/report.
  No ground-truth poses. See its `README.md`.
