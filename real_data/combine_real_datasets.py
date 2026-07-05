"""Combina varios datasets YOLO reales de banano con split GROUP-AWARE (sin fuga).

Los tiles se nombran como <imagen_fuente>_rN_cM.jpg. Si tiles de una misma imagen
fuente caen en train Y en test, el recall sale inflado (fuga). Este script agrupa por
imagen fuente y reparte GRUPOS COMPLETOS a train/val/test, de modo que ninguna imagen
fuente cruza splits — aunque venga de varios datasets (DS-v1, DS-v2, ...).

    python real_data/combine_real_datasets.py \
        --src DS-v1/ds-v1 DS-v2/ds-v2 --out combined --val 0.1 --test 0.1
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png")
TILE_RE = re.compile(r"^(.*?)(?:_r\d+_c\d+)?$")


def source_group(stem: str) -> str:
    """Prefijo de la imagen fuente: quita el sufijo _rN_cM del nombre del tile."""
    m = TILE_RE.match(stem)
    return m.group(1) if m else stem


def _bucket(group: str, val: float, test: float) -> str:
    """Asignacion determinista y estable de un grupo a train/val/test por hash."""
    h = int(hashlib.sha1(group.encode()).hexdigest(), 16) % 10_000 / 10_000.0
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def collect(src_roots):
    """Recorre las particiones existentes y junta (img_path, lbl_path, stem)."""
    items = []
    for root in src_roots:
        for split in ("train", "val", "valid", "test"):
            img_dir = os.path.join(root, "images", split)
            lbl_dir = os.path.join(root, "labels", split)
            if not os.path.isdir(img_dir):
                continue
            for name in os.listdir(img_dir):
                stem, ext = os.path.splitext(name)
                if ext.lower() not in IMG_EXT:
                    continue
                lbl = os.path.join(lbl_dir, stem + ".txt")
                items.append((os.path.join(img_dir, name), lbl, stem, ext.lower()))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="Raices de datasets YOLO")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--class-name", default="banano")
    args = ap.parse_args()

    items = collect(args.src)
    if not items:
        raise SystemExit("No se encontraron imagenes en las raices dadas.")

    # dedup por stem (misma tesela en v1 y v2 -> una sola)
    seen = {}
    for img, lbl, stem, ext in items:
        seen.setdefault(stem, (img, lbl, ext))

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    groups = set()
    for stem, (img, lbl, ext) in seen.items():
        grp = source_group(stem)
        groups.add(grp)
        split = _bucket(grp, args.val, args.test)
        shutil.copy2(img, os.path.join(args.out, "images", split, stem + ext))
        dst_lbl = os.path.join(args.out, "labels", split, stem + ".txt")
        if os.path.exists(lbl):
            # colapsa todas las clases a 0 = banano
            lines = []
            with open(lbl, encoding="utf-8") as fh:
                for line in fh:
                    p = line.split()
                    if len(p) >= 5:
                        p[0] = "0"
                        lines.append(" ".join(p))
            with open(dst_lbl, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
        else:
            open(dst_lbl, "w").close()
        counts[split] += 1

    yaml_path = os.path.join(args.out, "banano_combined.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"path: {os.path.abspath(args.out)}\n"
            "train: images/train\nval: images/val\ntest: images/test\n\n"
            f"nc: 1\nnames: ['{args.class_name}']\n"
        )
    print(f"Combinado: {sum(counts.values())} tiles unicos de {len(groups)} imagenes fuente")
    print(f"  train={counts['train']} val={counts['val']} test={counts['test']} (group-aware, sin fuga)")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
