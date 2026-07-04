# Architecture

BananaVision is built around four layers.

## 1. Image understanding

The default detector is `rgb-canopy`, a no-model baseline using contrast stretch, Excess Green, RGB triangular greenness, and green-dominance scoring. It is useful for first-pass counting, demos, dataset QA, and farms where banana canopy is visually separated from background vegetation.

The production detector is `yolo-seg`. It expects a trained segmentation model and can split a large predicted banana mat mask into likely individual plants using configured ground sample distance and crown size.

## 2. Instance separation

Banana clusters are handled after segmentation:

- connected canopy components are extracted;
- component area and spatial extent are compared with expected crown size;
- likely center peaks are selected with a minimum spacing rule using both visual score and distance-to-mask-edge peaks;
- one detection is emitted per likely plant.

This is intentionally parameterized because crown size changes with cultivar, age, altitude, and sensor.

Validation can label individuals inside the same banana mat with `group_id`,
`cluster_id`, or `mat_id`. Acceptance reports then calculate cluster recall,
fully detected cluster rate, under-split clusters, and over-split clusters so a
model cannot pass only because the aggregate count looks correct.
`cluster-review` turns those failures into per-mat JSON/CSV rows and optional
crops so labelers and model owners can fix the exact grouped banana mats that
failed.
`stratified-truth-coverage` checks that the locked annotations contain enough
truth support inside each claimed operating-domain stratum before model
performance is measured.
`stratified-acceptance` then re-gates the locked holdout by operating-domain
metadata such as farm, flight date, GSD band, and cultivar so a strong global
average cannot hide a failed field condition.

## 3. Geospatial output

If the source image has a world file or readable GeoTIFF transform, pixel centers are converted into map coordinates. EXIF GPS geotags are used as capture-level georeference evidence in preflight and mission QA, but not as a replacement for per-pixel plant coordinates. Outputs are written as CSV, JSON, and GeoJSON.
KML is also written for lon/lat detections so field teams can open results directly in Google Earth-style tools.
`geo-accuracy` compares predicted plant GeoJSON against field-truth point GeoJSON and reports match support, RMSE, p95 error, and recall before coordinates are trusted in production.

## 4. Deployment

Training and export are delegated to Ultralytics so the same model can be used as `.pt`, `.onnx`, or TensorRT `.engine`. The package exposes both CLI and FastAPI entry points.

## 5. Auditability

Each run records:

- BananaVision version;
- Python/platform;
- full inference config;
- config SHA-256;
- model path and model SHA-256 when a model file exists;
- per-image latency;
- mission-level duplicate suppression summary.

This makes field comparisons reproducible and gives operators evidence when a model is promoted or rejected.

## 6. Human Review Loop

The active-learning layer turns model uncertainty into work queues:

- `review-queue` finds low-confidence detections, zero-detection images, and dense split clusters;
- `review-crops` exports small image crops for fast human QA;
- corrected labels can be converted back to YOLO segmentation format;
- `split-dataset` prevents train/validation leakage by keeping farm/block/flight groups together;
- `tile-dataset` cuts large UAV imagery into trainable tiles and clips segmentation polygons;
- `quality-report` validates labels, splits, duplicate leakage, group leakage, and class counts;
- `cluster-benchmark` generates grouped synthetic banana mats and gates regression metrics before field runs;
- `validation-plan` calculates minimum holdout support for count-error and grouped-mat claims;
- `calibrate` searches confidence thresholds against field truth;
- `tune-config` searches banana mat splitting parameters against calibration truth;
- `field-report` bundles mission quality, prediction quality, flight-envelope, domain-shift, geospatial accuracy, truth coverage, stratified truth coverage, acceptance, stratified acceptance, benchmark, tuning, cluster-review, model, and run evidence into a static HTML artifact;
- `model-card` turns release evidence into a public model card with claim status and limitations;
- `evidence-manifest` indexes release evidence with paths, SHA256 hashes, reported statuses, and required-artifact checks;
- `release-audit` checks the complete publication package before a model is released.

## 7. Drone Runtime

Operational commands:

- `preflight` verifies readiness before flight;
- `flight-check` verifies planned GSD, overlap, and motion blur against the validated model envelope;
- `flight-log-audit` verifies actual post-flight telemetry/capture rows against that envelope;
- `capture-coverage` verifies capture logs against image files and flags missing frames, duplicate positions, and large capture gaps;
- `domain-profile` and `domain-check` compare mission imagery against the validated holdout visual domain;
- `validation-plan` turns operating-domain assumptions into auditable holdout sample thresholds;
- `deployment-smoke-test` runs real inference on the target machine and verifies smoke outputs before flight;
- `drone-ready` chains preflight, deployment smoke, deployment audit, and evidence indexing for routine final checks;
- `mission-quality` audits captured images for blur, exposure, resolution, and georeferencing;
- `mission-process` runs post-flight QA, inference, prediction QA, optional inventory update, and report generation;
- `mission-watch` processes new images from a watched camera folder;
- `prediction-quality` gates inference outputs before inventory or reporting;
- `mission-audit` combines mission run, capture QA, prediction QA, flight/domain evidence, and optional coordinate evidence before counts are delivered;
- `geo-accuracy` gates plant-level coordinates before GIS handoff or production release;
- `inventory-update` assigns stable plant IDs across missions;
- `inventory-diff` compares inventory snapshots and emits new/missing plant layers;
- `deploy-systemd` emits Linux services for mission-watch and API deployment;
- `holdout-lock` and `holdout-verify` freeze validation images and annotations before release metrics are produced;
- `truth-quality` verifies locked truth annotations do not contain duplicate centers, bad group IDs, oversized groups, or out-of-bounds centers;
- `truth-coverage` verifies the locked truth contains enough grouped banana mats for cluster-splitting claims;
- `stratified-truth-coverage` verifies truth support by farm, flight date, GSD band, cultivar, or other selected metadata strata before a broad claim is trusted;
- `acceptance-batch` can verify a holdout lock before inference, then validates with aggregate metrics, per-image metrics, sample support, and confidence intervals;
- `stratified-acceptance` checks the same acceptance report by farm, flight date, GSD band, cultivar, or other selected metadata strata;
- `promote-model` refuses models that lack passing acceptance or benchmark evidence;
- `release-audit` fails release packages with missing holdout verification, missing stratified truth coverage or stratified acceptance for 1% claims, missing flight-envelope evidence, missing domain-shift evidence, missing geospatial accuracy evidence, missing evidence, or unsupported claims;
- `evidence-manifest` produces the machine-readable release evidence index used for handoff and deployment packaging;
- `release-package` copies release artifacts into a hashed manifest and ZIP only after the release audit passes;
- `release-package-verify` recalculates package hashes from a folder, manifest, or ZIP and can require deployable model/config/evidence artifacts before installation;
- `deployment-audit` combines release-package verification, target preflight evidence, deployment smoke evidence, and deployment manifest checks before production flight;
- `drone-ready` writes the operational handoff bundle that proves those target-machine gates were run together;
- `publication-audit` verifies GitHub-facing project files, CI, security policy, operator docs, and release-flow documentation;
- `run_manifest.json` and `mission_watch_state.json` keep the mission auditable while it runs, including restart-safe processed-image state and failure records.
