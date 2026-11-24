"""Strict YAML config loading and hashing.

PyYAML implements YAML 1.1, where an exponent without a sign (e.g. `1.0e8`)
parses as a *string*. Silent string-for-number substitution in a frozen protocol
is unacceptable, so every numeric-looking string is rejected at load time.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

import yaml

NUMERIC_LIKE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


class ConfigError(ValueError):
    pass


def _walk(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


def load_config(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    bad = [(k, v) for k, v in _walk(cfg)
           if isinstance(v, str) and NUMERIC_LIKE.fullmatch(v)]
    if bad:
        raise ConfigError(
            "numeric-looking YAML scalars parsed as strings "
            "(add an explicit exponent sign, e.g. 1.0e+8): "
            + ", ".join(f"{k}={v!r}" for k, v in bad))
    return cfg


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(path: str | pathlib.Path) -> dict:
    """Return file sha256 plus canonical parsed-content sha256."""
    p = pathlib.Path(path)
    raw = p.read_bytes()
    cfg = load_config(p)
    return {
        "file": p.name,
        "config_version": cfg.get("config_version"),
        "sha256_file": hashlib.sha256(raw).hexdigest(),
        "sha256_canonical": hashlib.sha256(canonical_json(cfg).encode()).hexdigest(),
    }
