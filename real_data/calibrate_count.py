"""Calibracion HONESTA del conteo de banano + validacion cruzada.

Mide el ACIERTO DE CONTEO AGREGADO (inventario total del cultivo) del detector real
sobre imagenes que el modelo NUNCA vio (splits valid+test), con validacion cruzada de
K folds: en cada fold se fija el umbral de confianza y un factor de correccion usando
los folds de ajuste, y se mide el error de conteo en el fold retenido (out-of-fold).

Uso:
    python real_data/calibrate_count.py \
        --weights models/banana_real_v3.pt \
        --data-root RUTA/count_banana_plants \
        --out models/registry/real_v3_count_calibration.json

Reporta:
  - error de conteo OOF por fold y su media (acierto = 100 - error)
  - el punto de operacion global (conf, factor k) para produccion

Nota honesta: es acierto de CONTEO AGREGADO sobre un area, NO recall por planta
individual (que es ~0.80: en macollas densas hay plantas ocluidas en nadir). El total
acierta porque, sobre un area de densidad mixta, no-detecciones y falsos positivos se
compensan de forma estable. En un campo de densidad uniformemente extrema, recalibrar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def gt_count(lbl_dir: Path, stem: str) -> int:
    f = lbl_dir / (stem + ".txt")
    if not f.exists():
        return 0
    return sum(1 for ln in f.read_text().splitlines() if ln.strip())


def collect(root: Path, split: str, model, imgsz: int) -> list[tuple[int, list[float]]]:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    out = []
    res = model.predict(source=[str(p) for p in imgs], conf=0.001, iou=0.7, imgsz=imgsz,
                        device=0, verbose=False, stream=True)
    for p, r in zip(imgs, res):
        confs = sorted((r.boxes.conf.cpu().tolist() if r.boxes is not None else []), reverse=True)
        out.append((gt_count(lbl_dir, p.stem), confs))
    return out


def raw(cl: list[float], c: float) -> int:
    return sum(1 for x in cl if x >= c)


CONFS = [round(x, 2) for x in np.arange(0.10, 0.601, 0.02)]


def fit(items):
    best = (9e9, None, None)
    gt = np.array([g for g, _ in items], float)
    for c in CONFS:
        rc = np.array([raw(cl, c) for _, cl in items], float)
        if rc.sum() == 0:
            continue
        k = gt.sum() / rc.sum()
        err = abs((rc * k).sum() - gt.sum()) / gt.sum()
        score = err + 0.001 * abs(k - 1)   # ante empates, preferir menos correccion
        if score < best[0]:
            best = (score, c, k)
    return best[1], best[2]


def agg_err(items, c, k):
    rc = np.array([raw(cl, c) for _, cl in items], float)
    gt = np.array([g for g, _ in items], float)
    pred = rc * k
    return abs(pred.sum() - gt.sum()) / gt.sum() * 100, float(np.abs(pred - gt).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True, help="carpeta con {valid,test}/{images,labels}")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.data_root)
    model = YOLO(args.weights)
    items = []
    for s in args.splits:
        items += collect(root, s, model, args.imgsz)
    n = len(items)
    gt_total = int(sum(g for g, _ in items))
    print(f"imgs nunca vistas={n}  gt total={gt_total}")

    K = args.folds
    folds = [[items[i] for i in range(n) if i % K == f] for f in range(K)]
    per_fold = []
    print("\nfold | n | conf* | k* | error_conteo_OOF% | MAE")
    for f in range(K):
        held = folds[f]
        train = [it for g in range(K) if g != f for it in folds[g]]
        c, k = fit(train)
        e, m = agg_err(held, c, k)
        per_fold.append({"fold": f, "n": len(held), "conf": c, "k": round(k, 4),
                         "count_error_pct": round(e, 3), "mae": round(m, 3)})
        print(f"{f}   | {len(held)} | {c:.2f} | {k:.3f} | {e:6.2f} | {m:.2f}")

    errs = [p["count_error_pct"] for p in per_fold]
    mean_e, std_e = float(np.mean(errs)), float(np.std(errs))
    c, k = fit(items)
    print(f"\n== ERROR DE CONTEO OOF = {mean_e:.2f}% (std {std_e:.2f}) -> ACIERTO {100-mean_e:.2f}% ==")
    print(f"Punto de operacion produccion: conf={c}  factor_correccion={k:.4f}")

    report = {
        "metric": "acierto de conteo agregado (inventario) del cultivo de banano",
        "method": f"{K}-fold CV sobre {n} imagenes que el detector nunca vio (splits {args.splits})",
        "weights": Path(args.weights).name,
        "images_unseen": n,
        "gt_total": gt_total,
        "per_fold": per_fold,
        "count_error_oof_pct_mean": round(mean_e, 3),
        "count_error_oof_pct_std": round(std_e, 3),
        "count_accuracy_pct": round(100 - mean_e, 2),
        "operating_point": {"confidence_threshold": c, "count_correction_k": round(k, 4)},
        "honesty_note": ("Acierto de CONTEO AGREGADO sobre area, NO recall por planta (~0.80). "
                          "El total acierta por compensacion estable en densidad mixta; en campo "
                          "de densidad uniformemente extrema, recalibrar con imagenes locales."),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReporte -> {args.out}")


if __name__ == "__main__":
    main()
