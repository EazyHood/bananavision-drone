# Dataset Guide

## Capture protocol

- Use nadir or near-nadir UAV imagery.
- Prefer 70% front overlap and 70% side overlap for orthomosaics.
- Record altitude, camera model, focal length, image size, and GSD.
- Capture representative lighting: sunny, overcast, morning, afternoon.
- Include young, mature, sparse, dense, diseased, and replanted sections.
- Include negative examples: grass, weeds, roads, drainage, palms, mango, avocado, buildings.

## Annotation classes

Start with one class:

- `banana_plant`

For dense mats, annotate the visible individual plant/crown center as an instance mask where possible. If masks are too expensive, annotate boxes first, then use SAM/SAM2-assisted masks and human review.

## Annotation conversion

COCO instance annotations:

```bash
bananavision annotations coco annotations/instances_train.json dataset/labels/train --target-name banana_plant
```

LabelMe polygon or rectangle annotations:

```bash
bananavision annotations labelme annotations/labelme dataset/labels/train --target-label banana_plant
```

COCO polygons are converted to Ultralytics segmentation labels. COCO boxes without polygons are converted to rectangle polygons so a first detection model can still be trained.

## Split policy

Do not randomly split overlapping drone tiles from the same orthomosaic across train and validation. Split by field block, date, or flight. This avoids inflated validation scores.

Use group-aware splitting:

```bash
bananavision split-dataset raw/images raw/labels dataset_split \
  --manifest-csv raw/groups.csv \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1
```

The manifest CSV format is:

```csv
image,group
block-a/frame-001.jpg,farm-a-flight-2026-07-04
block-a/frame-002.jpg,farm-a-flight-2026-07-04
block-b/frame-001.jpg,farm-b-flight-2026-07-05
```

## Quality gate

Before training:

```bash
bananavision quality-report dataset_tiled/data.yaml --output runs/quality/quality_report.json
```

The report checks:

- missing splits;
- missing label files;
- invalid YOLO boxes or segmentation polygons;
- coordinates outside `[0, 1]`;
- unknown class IDs;
- duplicate image bytes across train/val/test;
- group leakage when `split_assignments.csv` exists;
- class counts and empty labels.

## Tiling orthomosaics

Large drone orthomosaics should be tiled after the train/val/test split:

```bash
bananavision tile-dataset dataset dataset_tiled --split train --tile-size 1024 --overlap 128
bananavision tile-dataset dataset dataset_tiled --split val --tile-size 1024 --overlap 128
```

The tiler clips YOLO segmentation polygons to tile boundaries and drops tiny clipped fragments below `--min-polygon-area-px`.

## Acceptance dataset

Plan the minimum holdout size before annotation:

```bash
bananavision validation-plan --output runs/validation/validation_plan.json
```

Use the generated `minimum_support` values as the floor for the locked holdout.

Before claiming production readiness, hold out at least:

- 3 farms or field blocks;
- 3 flight dates;
- 2 altitudes or GSD bands;
- 500 manually verified plant instances per condition where feasible.

For banana, the holdout must also annotate grouped mats with shared
`group_id`/`cluster_id`/`mat_id` values. Run `bananavision truth-quality` to
catch duplicate centers, singleton group IDs, oversized groups, and out-of-bounds
points. Then run `bananavision truth-coverage` and require enough grouped mats,
grouped plants, and cluster-containing images before using cluster-splitting
metrics for release. For multi-farm claims, also run
`bananavision stratified-truth-coverage` with metadata columns such as farm,
flight date, GSD band, and cultivar so weak conditions are visible before
training or acceptance.

The target for commercial release should include count error, precision, recall,
F1, cluster recall, duplicate rate, missed-plant rate, and geolocation error.

## Active learning loop

After inference:

```bash
bananavision review-queue runs/infer --output runs/active_learning/review_queue.json
bananavision review-crops runs/active_learning/review_queue.json --output runs/active_learning/review_crops
```

Review images with low confidence, zero detections, and dense split clusters first. Add corrected annotations back to the dataset, retrain, and rerun acceptance on a locked holdout set.
