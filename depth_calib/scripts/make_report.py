"""Assemble a Markdown report from an evaluation summary + per-image rows."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics


def fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = pathlib.Path(args.eval_dir)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(d / "per_image.csv", encoding="utf-8")))

    lines = ["# Evaluation report", ""]
    lines.append("## Main table (AbsRel, evaluated on valid GT mask; no per-image scaling)")
    lines.append("")
    lines.append("| System | n | AbsRel mean | 95% CI | by dataset |")
    lines.append("|---|---|---|---|---|")
    for sysname, s in summary.items():
        if sysname.startswith("_"):
            continue
        a = s.get("abs_rel", {})
        by = ", ".join(f"{k}={v['abs_rel_mean']:.4f}(n={v['n']})"
                       for k, v in sorted(s.get("by_dataset", {}).items()))
        lines.append(f"| {sysname} | {a.get('n','')} | {fmt(a.get('mean'))} | "
                     f"[{fmt(a.get('ci_low'))}, {fmt(a.get('ci_high'))}] | {by} |")

    deltas = summary.get("_paired_vs_B_raw", {})
    if deltas:
        lines += ["", "## Paired ΔAbsRel vs B_raw (positive = improvement)", "",
                  "| System | n pairs | Δ mean | 95% CI | improved/worse |",
                  "|---|---|---|---|---|"]
        for sysname, v in sorted(deltas.items()):
            dd = v["delta_abs_rel"]
            lines.append(f"| {sysname} | {v['n_pairs']} | {fmt(dd.get('mean'))} | "
                         f"[{fmt(dd.get('ci_low'))}, {fmt(dd.get('ci_high'))}] | "
                         f"{v['improved']}/{v['worse']} |")

    # coverage / fallback among calibrated systems
    lines += ["", "## Coverage and solver outcomes", "",
              "| System | rows | fallback | status counts |", "|---|---|---|---|"]
    for sysname in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == sysname]
        fb = [r for r in sub if str(r.get("fallback")).strip() not in ("", "0", "None")]
        statuses = {}
        for r in sub:
            st = str(r.get("solver_status") or r.get("status") or "")
            statuses[st] = statuses.get(st, 0) + 1
        lines.append(f"| {sysname} | {len(sub)} | {len(fb)} | "
                     f"{', '.join(f'{k}:{v}' for k, v in sorted(statuses.items()))} |")

    out = pathlib.Path(args.out) if args.out else (d / "report.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
