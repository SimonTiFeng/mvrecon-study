"""Verified, resumable file upload to a remote host over ssh/scp.

Splits the file locally with Python (reliable in Git Bash where dd's byte-skip is
unreliable), uploads each segment with size+sha256 verification and retries, then
concatenates remotely and verifies the whole-file hash.

Usage:
  python scripts/upload_verified.py LOCAL REMOTE_DIR REMOTE_NAME [--seg-mb 64]
        [--host aiserver@83.168.91.82] [--port 2222]
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys
import tempfile


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ssh(host: str, port: str, cmd: str, timeout: int = 120) -> str:
    r = subprocess.run(["ssh", "-o", "ServerAliveInterval=15", "-o", "ConnectTimeout=20",
                        "-p", port, host, cmd], capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "").strip()


def scp(local: pathlib.Path, host: str, port: str, remote: str) -> bool:
    r = subprocess.run(["scp", "-q", "-o", "ServerAliveInterval=15",
                        "-o", "ConnectTimeout=20", "-P", port,
                        str(local), f"{host}:{remote}"], capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("remote_dir")
    ap.add_argument("remote_name")
    ap.add_argument("--seg-mb", type=int, default=64)
    ap.add_argument("--host", default="aiserver@83.168.91.82")
    ap.add_argument("--port", default="2222")
    args = ap.parse_args()

    local = pathlib.Path(args.local)
    total = local.stat().st_size
    seg = args.seg_mb * 1024 * 1024
    nseg = (total + seg - 1) // seg
    whole = sha256_file(local)
    print(f"file={local} size={total} segments={nseg} sha256={whole}", flush=True)

    ssh(args.host, args.port, f"mkdir -p {args.remote_dir}/.parts")
    parts_dir = pathlib.Path(tempfile.mkdtemp(prefix="upseg_"))
    ok = 0
    with open(local, "rb") as f:
        for i in range(nseg):
            off = i * seg
            length = min(seg, total - off)
            f.seek(off)
            data = f.read(length)
            part = f"{args.remote_name}.seg{i:04d}"
            p = parts_dir / part
            p.write_bytes(data)
            lsize, lhash = len(data), hashlib.sha256(data).hexdigest()

            done = False
            for attempt in range(1, 6):
                rsize = ssh(args.host, args.port,
                            f"stat -c %s {args.remote_dir}/.parts/{part} 2>/dev/null || echo 0")
                if rsize == str(lsize):
                    rhash = ssh(args.host, args.port,
                                f"sha256sum {args.remote_dir}/.parts/{part} 2>/dev/null | cut -d' ' -f1")
                    if rhash == lhash:
                        done = True
                        break
                scp(p, args.host, args.port, f"{args.remote_dir}/.parts/{part}")
            if done:
                ok += 1
            print(f"seg {i}/{nseg-1} bytes={lsize} attempt={attempt} "
                  f"{'OK' if done else 'FAIL'}", flush=True)
            p.unlink(missing_ok=True)

    parts_dir.rmdir()
    print(f"verified segments: {ok}/{nseg}", flush=True)
    if ok != nseg:
        print("SEGMENTS_INCOMPLETE")
        return 1

    out = ssh(args.host, args.port,
              f"cd {args.remote_dir}/.parts && cat {args.remote_name}.seg* > ../{args.remote_name} "
              f"&& cd .. && rm -rf .parts && sha256sum {args.remote_name}",
              timeout=600)
    print("remote:", out)
    print("local :", whole)
    if not out.startswith(whole):
        print("HASH_MISMATCH")
        return 1
    print("UPLOAD_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
