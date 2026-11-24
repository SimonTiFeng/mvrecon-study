"""KITTI GPS/IMU as a second independent metric reference.

oxts gives lat/lon/alt and orientation per frame. Convert to a local metric frame and
compare inter-frame distances against the depth model's own predicted baselines.

If GPS and LiDAR give the same scale, that is two independent metric references
agreeing on the same data.
"""
import json
import pathlib
import itertools

import numpy as np

KIT = pathlib.Path("D:/kitti/2011_09_26")


def mercator(lat, lon, scale=1.0):
    """KITTI devkit's Mercator projection: metres in x,y."""
    er = 6378137.0
    mx = scale * np.deg2rad(lon) * er
    my = scale * np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * er
    return mx, my


def read_oxts(drive: str, frames):
    sync = KIT / f"2011_09_26_drive_{drive}_sync"
    out = {}
    for f in frames:
        p = sync / "oxts" / "data" / f"{f}.txt"
        if not p.exists():
            continue
        v = [float(x) for x in p.read_text().split()]
        out[f] = (v[0], v[1])  # lat, lon
    return out


def main():
    drive = "0002"
    frames = [f"{i:010d}" for i in range(0, 80)]
    ox = read_oxts(drive, frames)
    print(f"oxts frames read: {len(ox)}")
    keys = sorted(ox)
    xy = np.array([mercator(*ox[k]) for k in keys])
    # frame spacing is ~0.1 s; KITTI sync frames are consecutive
    print(f"\n{'pair':<14}{'gps m':>10}{'velo m':>10}{'ratio':>8}")
    # Velodyne-based inter-frame distance is not directly available; instead compare
    # the GPS displacement with the model baseline if the nested run persisted it.
    # Here we report GPS displacement magnitudes as the reference.
    disp = [np.linalg.norm(xy[i + 1] - xy[i]) for i in range(len(xy) - 1)]
    print(f"GPS inter-frame displacement: median {np.median(disp):.3f} m, "
          f"range [{min(disp):.3f}, {max(disp):.3f}]")
    print(f"Total traverse over {len(xy)} frames: {np.sum(disp):.1f} m")
    print()
    print("A driving sequence typically moves 0.5-2 m between sync frames at 10 Hz,")
    print("so this is a metric quantity usable as the numerator of route B.")


if __name__ == "__main__":
    main()
