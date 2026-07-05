"""Generate a synthetic banana dataset in YOLOv8-seg format.

Each mat becomes an instance polygon (class 0 = mat). Used to train the deep
learning model end to end and for the benchmark.

    python deep/make_synth_dataset.py --out dataset --train 160 --val 40 --size 640

NOTE: this is a SYNTHETIC dataset. The model trained on it demonstrates the full
pipeline (data -> training -> inference) and serves as a reproducible baseline,
but for real field use it must be relabeled with real images (see README / docs).
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
from skimage import measure, morphology

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from banano.synth import synth_plantation_labeled  # noqa: E402


def mask_to_polygon(mask, tol=1.5):
    """Convert the instance mask of ONE mat into a polygon (row, col).

    A mat can have several pseudostems (suckers) whose rosettes end up
    spatially separated -> the mask may be DISCONNECTED. The CONVEX HULL of the
    whole instance is taken so the polygon covers the full cluster (agronomic
    definition of a mat) and no rosettes are left unlabeled (which would produce
    false positives during training).
    """
    if mask.sum() < 25:  # too small
        return None
    hull = morphology.convex_hull_image(mask)
    padded = np.pad(hull.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    c = max(contours, key=len) - 1.0  # undo the pad
    c = measure.approximate_polygon(c, tolerance=tol)
    if len(c) < 3:
        return None
    return c


def write_tile(img, inst, img_path, lbl_path):
    H, W = inst.shape
    lines = []
    for mat_id in range(1, int(inst.max()) + 1):
        poly = mask_to_polygon(inst == mat_id)
        if poly is None:
            continue
        coords = []
        for r, c in poly:
            x = min(0.999999, max(0.0, c / W))
            y = min(0.999999, max(0.0, r / H))
            coords.append(f"{x:.6f} {y:.6f}")
        if len(coords) >= 3:
            lines.append("0 " + " ".join(coords))
    imageio.imwrite(img_path, img)
    with open(lbl_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def build(out, n_train, n_val, size, gsd):
    dirs = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(out, kind, split)
            os.makedirs(d, exist_ok=True)
            dirs[(kind, split)] = d

    rng = np.random.default_rng(12345)
    total_inst = 0
    plan = [("train", n_train, 0), ("val", n_val, 100000)]
    for split, n, base in plan:
        for i in range(n):
            seed = base + i
            spacing = float(rng.uniform(2.2, 3.0))
            img, inst, _, _, _ = synth_plantation_labeled(
                H=size, W=size, gsd_cm=gsd, spacing_m=spacing, seed=seed
            )
            name = f"{split}_{i:05d}"
            k = write_tile(
                img, inst,
                os.path.join(dirs[("images", split)], name + ".png"),
                os.path.join(dirs[("labels", split)], name + ".txt"),
            )
            total_inst += k
        print(f"  {split}: {n} tiles")

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"path: {os.path.abspath(out)}\n"
            "train: images/train\n"
            "val: images/val\n\n"
            "names:\n  0: mat\n"
        )
    print(f"Dataset ready in '{out}': {n_train}+{n_val} tiles, ~{total_inst} instances")
    print(f"data.yaml: {yaml_path}")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--train", type=int, default=160)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--gsd", type=float, default=3.0)
    args = ap.parse_args()
    build(args.out, args.train, args.val, args.size, args.gsd)


if __name__ == "__main__":
    main()
