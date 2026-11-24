#!/usr/bin/env python
"""Build the package into a staging dir and swap it into place.

Works around a Windows lock on the package directory itself (some shell's cwd): only
the contents are cleared, and the data directory is moved rather than copied.
The package dir is temporarily emptied, so no shell may sit inside it.
"""
from __future__ import annotations

import shutil

import build_package as bp

PKG = bp.PKG
STAGE = PKG.parent / '_stage_mvrecon-workbench'


def clear(d):
    if not d.exists():
        return
    for child in d.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError:
            print('locked, skipping:', child)


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE, ignore_errors=True)
    bp.set_pkg(STAGE)
    bp.main()
    clear(PKG)
    PKG.mkdir(exist_ok=True)
    for child in STAGE.iterdir():
        child.rename(PKG / child.name)
    shutil.rmtree(STAGE, ignore_errors=True)
    total = sum(f.stat().st_size for f in PKG.rglob('*') if f.is_file())
    print('package in place:', PKG, 'MB %.1f' % (total / 1e6))


if __name__ == '__main__':
    main()
