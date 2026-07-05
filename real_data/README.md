# Real data and banana model training

This directory contains the workflow to (re)train the model with **real UAV imagery**.
The included model (`models/banana_real_v1.pt`) was trained with this process.

## Real dataset used

**AI-BananaMapping DS-v1** — real RGB UAV imagery (nadir, low altitude) of banana
crops, tiled to 1024 px, with per-plant YOLO annotations.
- Source: Zenodo [record 20945958](https://zenodo.org/records/20945958), **CC-BY-4.0 license**.
- ~14,180 train / 4,645 val / 4,611 test. Single class: banana.

### Download and prepare

```bash
# 1. Download (9.8 GB, no login)
curl -L -o ds-v1.rar "https://zenodo.org/api/records/20945958/files/ds-v1.rar/content"
7z x ds-v1.rar          # requires 7-Zip / unrar

# 2. (optional) collapse classes to "banana" if your dataset has multiple disease classes
python real_data/prepare_real_dataset.py --root DS-v1/ds-v1 --out DS-v1/banana_yolo
```

The `data.yaml` must point (absolute path) to `images/{train,val,test}` with `names: [banana]`.

## Train

```bash
pip install -e ".[ml]"
# YOLOv8 detection from COCO weights, on the real imagery:
yolo detect train model=yolov8s.pt data=DS-v1/ds-v1/banana_real.yaml epochs=25 imgsz=640 batch=24
```

> ⚠️ **On Windows**, train from a `.py` file or the `yolo` CLI (NOT `python - <<PY`
> via stdin) or the DataLoader multiprocessing fails with `OSError: '<stdin>'`.

## Evaluate on the real test set and register

```bash
python real_data/eval_real.py --weights runs/detect/train/weights/best.pt \
    --data DS-v1/ds-v1/banana_real.yaml --out real_eval
bananavision register-model runs/detect/train/weights/best.pt "real-v1.1" \
    --config configs/banana_real_model.yaml --acceptance-report real_eval/real_test_metrics.json
```

## Scale and fine-tuning

The dataset is **low altitude (plant scale)**. For high-altitude orthomosaics,
crop tiles to your GSD and **fine-tune** (`resume`/transfer learning) with a few images from
your farm. The framework includes a *group-aware* split (no farm/block/flight leakage),
`bananavision audit-dataset`, calibration, and *active learning* queues to speed it up.
