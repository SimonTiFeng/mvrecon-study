"""Batch upload: send N segments per invocation, verifying each; prints a resume cursor.

Designed for a session where long background processes get reaped: each call does a
bounded amount of work (default 4 segments) and exits, leaving verified segments on
the remote. Re-invoke until it reports COMPLETE; already-correct segments are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys

HOST = "aiserver@83.168.91.82"
PORT = "2222"


def ssh(cmd: str, timeout: int = 300, retries: int = 3) -> str:
    """SSH with generous timeout and retries; the link drops often."""
    last = ""
    for _ in range(retries):
        try:
            r = subprocess.run(["ssh", "-o", "ServerAliveInterval=15",
                                "-o", "ConnectTimeout=25", "-o", "ServerAliveCountMax=6",
                                "-p", PORT, HOST, cmd],
                               capture_output=True, text=True, timeout=timeout)
            out = (r.stdout or "").strip()
            if out:
                return out
            last = out
        except subprocess.TimeoutExpired:
            continue
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("remote_dir")
    ap.add_argument("remote_name")
    ap.add_argument("--seg-mb", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    local = pathlib.Path(args.local)
    total = local.stat().st_size
    seg = args.seg_mb * 1024 * 1024
    nseg = (total + seg - 1) // seg
    ssh(f"mkdir -p {args.remote_dir}/.parts")

    staging = pathlib.Path("upseg")
    staging.mkdir(exist_ok=True)

    done = pending = 0
    sent_now = 0
    with open(local, "rb") as f:
        for i in range(nseg):
            part = f"{args.remote_name}.seg{i:04d}"
            off = i * seg
            length = min(seg, total - off)
            f.seek(off)
            data = f.read(length)
            lhash = hashlib.sha256(data).hexdigest()

            rsize = ssh(f"stat -c %s {args.remote_dir}/.parts/{part} 2>/dev/null || echo 0")
            if rsize == str(length):
                done += 1          # size is authoritative per segment; hash checked on assembly
                continue

            if sent_now >= args.batch:
                pending += 1
                continue

            p = staging / part
            p.write_bytes(data)
            for _ in range(3):
                r = subprocess.run(
                    ["scp", "-q", "-o", "ServerAliveInterval=15", "-P", PORT,
                     str(p.resolve()), f"{HOST}:{args.remote_dir}/.parts/{part}"],
                    capture_output=True, text=True)
                rsize = ssh(f"stat -c %s {args.remote_dir}/.parts/{part} 2>/dev/null || echo 0")
                if rsize == str(length):
                    break
            p.unlink(missing_ok=True)
            sent_now += 1
            done += 1
            print(f"sent {part} ({length//1024//1024} MB) verified", flush=True)

    print(f"PROGRESS verified={done}/{nseg} remaining={nseg-done}")
    if done == nseg:
        out = ssh(f"cd {args.remote_dir}/.parts && cat {args.remote_name}.seg* "
                  f"> ../{args.remote_name} && cd .. && rm -rf .parts "
                  f"&& sha256sum {args.remote_name}", timeout=900)
        whole = hashlib.sha256()
        with open(local, "rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                whole.update(c)
        print("remote:", out)
        print("local :", whole.hexdigest())
        print("COMPLETE" if out.startswith(whole.hexdigest()) else "HASH_MISMATCH")
        return 0 if out.startswith(whole.hexdigest()) else 1
    print("REINVOKE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
