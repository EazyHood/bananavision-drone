# Operator Manual

This manual is the field sequence for a BananaVision mission. It assumes the
model and config have already passed validation for the target farm, sensor, GSD,
altitude band, and banana cultivar.

## 1. Before Flight

Confirm the software and model are installed:

```bash
bananavision preflight \
  --input /data/mission/incoming \
  --output runs/preflight/preflight_report.json \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.pt
```

Confirm the planned capture is inside the validated flight envelope:

```bash
bananavision flight-check \
  --output runs/flight_check/flight_check_report.json \
  --config configs/banana_uav.yaml \
  --gsd-cm 2.0 \
  --front-overlap 75 \
  --side-overlap 72 \
  --speed-mps 4 \
  --exposure-ms 4
```

Run a target-machine smoke image through the installed runtime:

```bash
bananavision deployment-smoke-test /data/smoke/banana_smoke.jpg \
  --output runs/deployment_smoke/deployment_smoke_report.json \
  --artifacts-dir runs/deployment_smoke/artifacts \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.pt \
  --min-detections 1
```

For routine operation, use the combined final gate:

```bash
bananavision drone-ready /models/banana-v1 \
  edge/systemd/deployment_manifest.json \
  /data/smoke/banana_smoke.jpg \
  --output runs/drone_ready \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.pt \
  --min-detections 1
```

Do not fly for production counts if any command fails. Fix the model path,
storage, GSD, overlap, speed, exposure, mission plan, or runtime installation
first.

## 2. During Flight

For companion computers that receive image files during flight:

```bash
bananavision mission-watch /data/mission/incoming \
  --output /data/mission/output \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.pt
```

The watcher resumes from `mission_watch_state.json` after restart. Use
`--no-resume` only for deliberate reprocessing.

## 3. After Capture

Audit capture quality:

```bash
bananavision mission-quality /data/mission/incoming \
  --output runs/mission_quality/mission_quality_report.json \
  --require-georef
```

Compare imagery against the validated holdout visual domain:

```bash
bananavision domain-check /data/mission/incoming runs/domain/domain_profile.json \
  --output runs/domain/domain_check_report.json
```

Audit the actual flight/capture log when available:

```bash
bananavision flight-log-audit /data/mission/flight_log.csv \
  --output runs/flight_log/flight_log_audit.json \
  --config configs/banana_uav.yaml
```

Run the full post-flight workflow:

```bash
bananavision mission-process /data/mission/incoming \
  --output runs/mission_process \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.pt \
  --inventory-dir farm_inventory
```

## 4. Before Reporting Counts

Run prediction QA:

```bash
bananavision prediction-quality runs/mission_process/infer \
  --output runs/prediction_quality/prediction_quality_report.json
```

Review low-confidence detections, edge detections, duplicate-risk detections, and
dense-cluster splits before publishing plant counts or updating inventory.

Run the mission delivery audit:

```bash
bananavision mission-audit \
  runs/mission_process/infer/run_manifest.json \
  runs/mission_process/mission_quality/mission_quality_report.json \
  runs/mission_process/prediction_quality/prediction_quality_report.json \
  --flight-check-report runs/flight_check/flight_check_report.json \
  --flight-log-report runs/flight_log/flight_log_audit.json \
  --capture-coverage-report runs/capture_coverage/capture_coverage_report.json \
  --require-capture-coverage \
  --domain-check-report runs/domain/domain_check_report.json \
  --output runs/mission_audit/mission_audit.json
```

If the output will be used as plant coordinates in GIS or inventory systems,
validate geospatial accuracy against field-truth GeoJSON:

```bash
bananavision geo-accuracy runs/mission_process/infer/mission.detections.geojson /data/holdout/plants.truth.geojson \
  --output runs/geo_accuracy/geo_accuracy_report.json \
  --tolerance-m 1 \
  --max-rmse-m 1 \
  --max-p95-m 1.5 \
  --min-recall 0.99
```

## 5. Release Evidence

A production model handoff needs:

- locked holdout verification;
- passing truth-quality report;
- passing truth-coverage report;
- passing stratified-truth-coverage report for all claimed operating strata;
- passing acceptance-batch report;
- passing stratified-acceptance report for all claimed operating strata;
- cluster-annotated banana mat support in the acceptance report;
- cluster-review report when grouped-mat gates fail during calibration or remediation;
- target hardware benchmark;
- mission-quality report;
- prediction-quality report;
- mission-audit report;
- flight-check report;
- flight-log-audit report when a telemetry/capture log is available;
- capture-coverage report when image names and capture positions are available;
- domain-check report;
- geo-accuracy report;
- model card;
- field report;
- passing release-audit report;
- passing evidence-manifest report with all release-critical artifacts required;
- passing deployment-smoke-test report from the target drone computer;
- passing drone-ready summary for routine operational handoff;
- release-package ZIP and manifest.
- passing `release-package-verify --require-deployment-artifacts` report after copying the package to the deployment machine.
- passing `deployment-audit` report from the deployed package, drone preflight report, and deployment manifest.

Do not claim a 1% count-error rate unless `release-audit` passes with the 1%
thresholds, stratified truth coverage and stratified acceptance passed for all
claimed field conditions, cluster gates were applied for grouped banana mats,
and the package was built without `--allow-failed-audit`.
