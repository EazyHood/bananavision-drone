# Training

## Recommended path

1. Build an Ultralytics segmentation dataset.
2. Convert COCO/LabelMe annotations if needed.
3. Split by farm/block/flight with `split-dataset`.
4. Tile large UAV imagery after splitting.
5. Run `bananavision audit-dataset`.
6. Train a small segmentation model first.
7. Inspect false positives, false negatives, and active-learning queue items.
8. Add hard negatives and clustered mats.
9. Export to ONNX/TensorRT only after validation passes.

## Commands

```bash
bananavision audit-dataset dataset/data.yaml
bananavision annotations coco annotations/instances_train.json dataset/labels/train --target-name banana_plant
bananavision split-dataset raw/images raw/labels dataset_split --manifest-csv raw/groups.csv
bananavision tile-dataset dataset_split dataset_tiled --split train --tile-size 1024 --overlap 128
bananavision tile-dataset dataset_split dataset_tiled --split val --tile-size 1024 --overlap 128
bananavision quality-report dataset_tiled/data.yaml --output runs/quality/quality_report.json
bananavision train dataset_tiled/data.yaml --model yolo26n-seg.pt --epochs 160 --imgsz 1024 --batch 8 --device 0
bananavision validate dataset/data.yaml runs/banana/seg/weights/best.pt --imgsz 1024 --device 0
bananavision review-queue runs/infer --output runs/active_learning/review_queue.json
bananavision review-crops runs/active_learning/review_queue.json --output runs/active_learning/review_crops
```

## Important knobs

- `gsd_cm`: centimeters per pixel in the orthomosaic/frame.
- `expected_crown_diameter_m`: average visible crown diameter.
- `min_center_distance_ratio`: how close two predicted plant centers may be.
- `center_distance_weight`: how much instance splitting trusts distance-to-mask-edge peaks when score maps are flat.
- `canopy_fill_ratio`: expected mask fill compared with a full circular crown.
- `max_split_instances`: upper bound for plants split from one connected mat component.

These knobs matter because banana plants often appear as grouped mats rather than isolated tree crowns.

## Calibration

Before using field data, run a local grouped-mat smoke scene:

```bash
bananavision synthetic \
  --image examples/clustered_smoke.jpg \
  --truth examples/clustered_smoke.truth.json \
  --plants 12 \
  --clustered-mats 2 \
  --min-plants-per-mat 3 \
  --max-plants-per-mat 3

bananavision truth-coverage examples/clustered_smoke.truth.json \
  --min-cluster-count 2 \
  --min-cluster-truth-count 6

bananavision cluster-benchmark --output runs/cluster_benchmark
```

For real field holdouts, add `bananavision stratified-truth-coverage` with the
holdout metadata CSV before training decisions are treated as release evidence.

After running inference on a validation image with field truth:

```bash
bananavision calibrate runs/infer/block-a.detections.json truth/block-a.truth.json --tolerance-px 24
```

Use calibration to choose a confidence threshold for the operating domain. When truth
points include `group_id`, calibration also prefers thresholds that preserve banana
cluster recall. Keep the final acceptance set locked and separate from calibration
data.

## Banana Cluster Tuning

Banana plants often appear as grouped mats. Tune the individual-splitting parameters on a calibration block:

```bash
bananavision tune-config calibration/block-a.tif calibration/block-a.truth.json \
  --config configs/banana_uav.yaml \
  --output runs/tuning/block-a.tuning.json \
  --output-config runs/tuning/block-a.tuned.yaml \
  --tolerance-px 24 \
  --center-distance-weight 0.2 --center-distance-weight 0.35 --center-distance-weight 0.6 \
  --crown-m 1.8 --crown-m 2.2 --crown-m 2.6 \
  --min-distance-ratio 0.35 --min-distance-ratio 0.42 --min-distance-ratio 0.50 \
  --canopy-fill-ratio 0.48 --canopy-fill-ratio 0.58 --canopy-fill-ratio 0.68 \
  --max-split-instances 8 --max-split-instances 12 --max-split-instances 16
```

Annotate grouped banana mats in calibration and holdout truth with a shared
`group_id`, `cluster_id`, or `mat_id` per mat:

```json
{
  "centers": [
    { "x": 123.4, "y": 567.8, "group_id": "mat-001" },
    { "x": 142.0, "y": 573.1, "group_id": "mat-001" },
    { "x": 166.8, "y": 562.4, "group_id": "mat-001" }
  ]
}
```

Then run acceptance with `block-a.tuned.yaml` on a different holdout block and set
`--min-cluster-recall`, `--min-cluster-full-detection-rate`, and
`--min-cluster-count`. Do not tune on the final acceptance set.

When cluster metrics fail, generate an actionable grouped-mat review:

```bash
bananavision cluster-review runs/infer/block-a.detections.json calibration/block-a.truth.json \
  --output runs/cluster_review/block-a.cluster_review.json \
  --tolerance-px 24 \
  --crops-dir runs/cluster_review/crops
```

The report lists each grouped mat, whether it was under-split or over-split,
missing truth centers, extra predictions near the mat, and crop boxes for human
QA.

## Promotion

Once a model passes acceptance and benchmark gates, register it:

```bash
bananavision register-model runs/banana/seg/weights/best.pt banana-v1 \
  --config configs/banana_uav.yaml \
  --acceptance-report runs/acceptance/acceptance_report.json \
  --benchmark-report runs/benchmark/benchmark_report.json
```
