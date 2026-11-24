"""Stage a manifest's images onto HPC2 and emit a remote manifest.

Local processed datasets -> /hpc2hdd/home/hourujiang/jhspoolers/depthcalib/data/<dataset>/
Remote manifest rows keep all fields but point dataset_root at the remote data root.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tarfile
import tempfile

REMOTE = "/hpc2hdd/home/hourujiang/jhspoolers/depthcalib"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="remote manifest basename, e.g. pilot")
    ap.add_argument("--remote", default=REMOTE)
    args = ap.parse_args()

    rows = []
    for m in args.manifest:
        rows += [json.loads(l) for l in pathlib.Path(m).read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    by_ds: dict[str, list[tuple[pathlib.Path, str]]] = {}
    remote_rows = []
    for r in rows:
        src = pathlib.Path(r["dataset_root"]) / r["image"]
        rel = f"{r['dataset']}/{r['image']}"
        by_ds.setdefault(r["dataset"], []).append((src, rel))
        rr = dict(r)
        rr["dataset_root"] = f"{args.remote}/data"
        rr["image"] = rel
        remote_rows.append(rr)

    subprocess.run(["ssh", "-o", "BatchMode=yes", "hpc2",
                    f"mkdir -p {args.remote}/data {args.remote}/manifests {args.remote}/cache/depths {args.remote}/logs"],
                   check=True)
    for ds, files in by_ds.items():
        with tempfile.TemporaryDirectory() as td:
            tar_path = pathlib.Path(td) / f"{ds}.tar"
            with tarfile.open(tar_path, "w") as tf:
                seen = set()
                for src, rel in files:
                    if rel in seen:
                        continue
                    seen.add(rel)
                    tf.add(src, arcname=f"{ds}/{pathlib.Path(rel).relative_to(ds)}")
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(tar_path),
                            f"hpc2:{args.remote}/{ds}.tar"], check=True)
        subprocess.run(["ssh", "-o", "BatchMode=yes", "hpc2",
                        f"tar xf {args.remote}/{ds}.tar -C {args.remote}/data && rm -f {args.remote}/{ds}.tar"],
                       check=True)
        print(f"[{ds}] {len(files)} files staged")

    with tempfile.TemporaryDirectory() as td:
        mf = pathlib.Path(td) / f"{args.name}.jsonl"
        mf.write_text("\n".join(json.dumps(r) for r in remote_rows) + "\n")
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(mf),
                        f"hpc2:{args.remote}/manifests/{args.name}.jsonl"], check=True)
    print(f"remote manifest: {args.remote}/manifests/{args.name}.jsonl "
          f"({len(remote_rows)} rows)")


if __name__ == "__main__":
    main()
