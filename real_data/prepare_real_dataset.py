"""Prepares a real banana YOLO dataset to TRAIN A PLANT DETECTOR.

Many real aerial banana datasets are labeled by DISEASE (e.g.
healthy / Moko / Sigatoka). For a plant counter, we collapse all classes
into a single one: "banana". This script:

  1. Locates the splits (train/val/test) and their images + YOLO labels.
  2. Rewrites each label setting the class index = 0 (single class).
  3. Writes a new data.yaml with names: {0: banana}.

It does not copy images (it uses the existing ones); it only creates collapsed labels and the yaml.

    python deep/prepare_real_dataset.py --root realdata/DS-v1 --out realdata/banana_yolo
"""
from __future__ import annotations

import argparse
import os
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_splits(root):
    """Returns dict split -> (images_dir, labels_dir) by detecting the YOLO structure."""
    splits = {}
    # pattern 1: root/images/train, root/labels/train
    for split in ("train", "val", "valid", "test"):
        img = os.path.join(root, "images", split)
        lbl = os.path.join(root, "labels", split)
        if os.path.isdir(img) and os.path.isdir(lbl):
            splits[split] = (img, lbl)
    if splits:
        return splits
    # pattern 2: root/train/images, root/train/labels
    for split in ("train", "val", "valid", "test"):
        img = os.path.join(root, split, "images")
        lbl = os.path.join(root, split, "labels")
        if os.path.isdir(img) and os.path.isdir(lbl):
            splits[split] = (img, lbl)
    return splits


def collapse_labels(src_lbl_dir, dst_lbl_dir):
    os.makedirs(dst_lbl_dir, exist_ok=True)
    n_files = n_boxes = 0
    for name in os.listdir(src_lbl_dir):
        if not name.endswith(".txt"):
            continue
        out_lines = []
        with open(os.path.join(src_lbl_dir, name), encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                parts[0] = "0"  # collapse the class to 0 = banana
                out_lines.append(" ".join(parts))
                n_boxes += 1
        with open(os.path.join(dst_lbl_dir, name), "w", encoding="utf-8") as fh:
            fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        n_files += 1
    return n_files, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root of the extracted YOLO dataset")
    ap.add_argument("--out", required=True, help="Output folder (collapsed labels + yaml)")
    args = ap.parse_args()

    splits = find_splits(args.root)
    if not splits:
        raise SystemExit(
            f"No YOLO structure (images/labels) found under {args.root}. "
            "Check the extracted structure."
        )

    os.makedirs(args.out, exist_ok=True)
    yaml_splits = {}
    for split, (img_dir, lbl_dir) in splits.items():
        dst_lbl = os.path.join(args.out, "labels", split)
        nf, nb = collapse_labels(lbl_dir, dst_lbl)
        # we link the original images via absolute path in the yaml
        yaml_splits[split] = (os.path.abspath(img_dir), os.path.abspath(dst_lbl))
        print(f"  {split}: {nf} labels, {nb} boxes -> single class 'banana'")

    # ultralytics expects labels to be next to the images or via 'path'.
    # We create a standard images/labels structure per split with symlink/copy of images.
    data_root = os.path.join(args.out, "data")
    for split, (img_dir, dst_lbl) in yaml_splits.items():
        di = os.path.join(data_root, "images", split)
        dl = os.path.join(data_root, "labels", split)
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        # copy labels
        for n in os.listdir(dst_lbl):
            shutil.copy2(os.path.join(dst_lbl, n), os.path.join(dl, n))
        # copy images (same basename)
        for n in os.listdir(img_dir):
            if n.lower().endswith(IMG_EXT):
                shutil.copy2(os.path.join(img_dir, n), os.path.join(di, n))

    val_key = "val" if "val" in splits else ("valid" if "valid" in splits else "test")
    train_key = "train" if "train" in splits else list(splits)[0]
    yaml_path = os.path.join(args.out, "banana_real.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"path: {os.path.abspath(data_root)}\n"
            f"train: images/{train_key}\n"
            f"val: images/{val_key}\n"
        )
        if "test" in splits:
            fh.write("test: images/test\n")
        fh.write("\nnames:\n  0: banana\n")
    print(f"\ndata.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
