"""Shared geometry types. All x/y are original-image pixels, pixel centers at integers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AnchorGeom:
    anchor_id: str
    object_group_id: str
    knowledge_group_id: str
    p1: np.ndarray            # (2,) float, original image px
    p2: np.ndarray
    q1: np.ndarray            # (3,) ray, z=1
    q2: np.ndarray
    length_m: float
    length_interval_m: Optional[tuple] = None
    d1: Optional[float] = None  # backend z-depth at p1 (resampled), if affine backend
    d2: Optional[float] = None
    r1: Optional[float] = None  # inverse-depth/disparity at p1, if relative backend
    r2: Optional[float] = None
    quality_flags: tuple = ()


def group_weights(anchor_group_ids: list[str], cap: float = 1.0) -> np.ndarray:
    """Weight so each group contributes at most `cap` in total."""
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(anchor_group_ids):
        groups.setdefault(g, []).append(i)
    w = np.zeros(len(anchor_group_ids), dtype=float)
    for idx in groups.values():
        w[idx] = cap / len(idx)
    return w
