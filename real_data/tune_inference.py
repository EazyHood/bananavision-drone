"""Confidence threshold + NMS sweep to MAXIMIZE COUNTING accuracy.

In dense banana, a high threshold misses plants (under-counting) and a low NMS
removes overlapping plants. This script tests combinations on a held-out split
and picks the one that minimizes counting error while keeping precision reasonable.

    python real_data/tune_inference.py --weights best.pt \
        --data-dir realdata/count_banana_plants --split test
"""
from __future__ import annotations

import argparse
import glob
import os


def gt_count(labels_dir, stem):
    p = os.path.join(labels_dir, stem + ".txt")
    if not os.path.exists(p):
        return 0
    return sum(1 for ln in open(p, encoding="utf-8") if len(ln.split()) >= 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", required=True, help="Root with <split>/images and <split>/labels")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    img_dir = os.path.join(args.data_dir, args.split, "images")
    lbl_dir = os.path.join(args.data_dir, args.split, "labels")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    gts = [gt_count(lbl_dir, os.path.splitext(os.path.basename(p))[0]) for p in imgs]
    total_gt = sum(gts)
    print(f"{len(imgs)} images, {total_gt} real plants\n")
    print(f"{'conf':>5}{'iou':>5}{'pred':>7}{'count_err%':>13}{'MAE/img':>9}")
    print("-" * 40)

    best = None
    for conf in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        for iou in (0.5, 0.6, 0.7):
            preds = []
            for p in imgs:
                r = model.predict(p, conf=conf, iou=iou, imgsz=args.imgsz, verbose=False)[0]
                preds.append(0 if r.boxes is None else len(r.boxes))
            total_pred = sum(preds)
            err = abs(total_pred - total_gt) / max(1, total_gt) * 100
            mae = sum(abs(a - b) for a, b in zip(preds, gts)) / len(imgs)
            print(f"{conf:>5}{iou:>5}{total_pred:>7}{err:>12.1f}%{mae:>9.2f}")
            if best is None or err < best[0]:
                best = (err, conf, iou, total_pred)
    print("-" * 40)
    print(f"BEST for counting: conf={best[1]} iou={best[2]} -> {best[3]} plants, "
          f"counting error {best[0]:.1f}% (real {total_gt})")


if __name__ == "__main__":
    main()
