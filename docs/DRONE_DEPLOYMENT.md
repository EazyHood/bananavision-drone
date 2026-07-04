# Drone Deployment

## Edge target

Recommended minimum:

- NVIDIA Jetson Orin Nano / Orin NX, or equivalent edge GPU;
- 8 GB RAM or more;
- local SSD for image buffering;
- Ubuntu LTS with CUDA/TensorRT support for `.engine` models.

## Runtime modes

1. Post-flight orthomosaic mode:
   - highest accuracy;
   - easiest georeferencing;
   - best for counts and maps.

2. In-flight frame mode:
   - lower latency;
   - requires frame de-duplication across overlap;
   - best for scouting and mission decisions.

## Preflight

Run this before a drone mission:

```bash
bananavision preflight \
  --input /data/mission/incoming \
  --output runs/preflight/preflight_report.json \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.pt
```

The report checks Python, config validity, model file, required dependencies, writable output path, free disk space, GPU visibility, and georeferencing on the first image when available. It accepts world files, GeoTIFF transforms, and EXIF GPS geotags.

## Flight envelope

Run this before takeoff to confirm the planned flight matches the model's
validated GSD and capture assumptions:

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

If the GSD is not measured directly, estimate it from altitude and camera
geometry:

```bash
bananavision flight-check \
  --altitude-m 60 \
  --sensor-width-mm 13.2 \
  --focal-length-mm 8.8 \
  --image-width-px 5472 \
  --front-overlap 75 \
  --side-overlap 72
```

The report fails when GSD drift, front overlap, side overlap, or computed motion
blur falls outside the configured envelope. Keep this report with the mission
evidence and pass it to `release-audit`.

After the mission, audit the actual telemetry/capture CSV if available:

```bash
bananavision flight-log-audit /data/mission/flight_log.csv \
  --output runs/flight_log/flight_log_audit.json \
  --config configs/banana_uav.yaml
```

The CSV can contain measured `gsd_cm`, `front_overlap`, `side_overlap`,
`speed_mps`, and `exposure_ms`, or altitude and camera-geometry columns. This
post-flight evidence catches real drift that the preflight plan could not see.

When the drone stack writes image names and positions to a capture log, audit
coverage as well:

```bash
bananavision capture-coverage /data/mission/capture_log.csv \
  --images /data/mission/incoming \
  --output runs/capture_coverage/capture_coverage_report.json \
  --require-image-files \
  --max-position-gap-m 35
```

This catches missing frames, repeated capture positions, and large jumps that
can create uncovered strips even when per-image inference looks healthy.

## Visual domain check

Create a domain profile from the locked/validated holdout images:

```bash
bananavision domain-profile /data/holdout/images \
  --output runs/domain/domain_profile.json
```

Before trusting a new mission, compare its images against that profile:

```bash
bananavision domain-check /data/mission/incoming runs/domain/domain_profile.json \
  --output runs/domain/domain_check_report.json \
  --max-outlier-fraction 0
```

The check uses RGB histograms plus color, luma, saturation, and resolution
features to catch camera, processing, lighting, or crop-state shifts that can
invalidate a model even when flight altitude is correct.

## Mission image quality

Run this after capture or during post-flight intake:

```bash
bananavision mission-quality /data/mission/incoming \
  --output runs/mission_quality/mission_quality_report.json \
  --min-width 1024 \
  --min-height 768 \
  --require-georef
```

The report checks each image for resolution, blur/focus score, underexposure, overexposure, saturation, and georeferencing. EXIF GPS geotags are accepted for capture QA, but plant-level map coordinates still need an orthomosaic, GeoTIFF/world-file transform, or equivalent camera geometry. Failed captures should be reflown or removed from acceptance metrics instead of being treated as model failures.

## One-command post-flight processing

For routine operation, run the capture QA, inference, prediction QA, optional inventory update, and HTML report in one command:

```bash
bananavision mission-process /data/mission/incoming \
  --output runs/mission_process \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.pt \
  --require-georef \
  --inventory-dir farm_inventory
```

The command writes `mission_process_manifest.json`, `field_report.html`, `mission_quality/`, `infer/`, and `prediction_quality/`. If a quality gate fails, it exits non-zero unless `--no-fail` is supplied for exploratory runs.

## Mission watch

Use this on a companion computer when images are written to a local folder:

```bash
bananavision mission-watch /data/mission/incoming \
  --output runs/mission \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.pt
```

The watcher reuses the detector in memory, waits for files to settle, writes per-image outputs, updates `mission.detections.*`, and keeps `mission_watch_state.json` plus `run_manifest.json` current.

If the companion computer restarts, run the same command again. The watcher resumes from `mission_watch_state.json` and skips already processed images. Use `--no-resume` only for deliberate reprocessing.

## Companion integrations

For MAVLink-based stacks that already write camera frames to a companion-computer
folder, `edge/mavlink/mission_counter.py` wraps the same restart-safe watcher:

```bash
python edge/mavlink/mission_counter.py \
  --watch /data/mission/incoming \
  --output /data/mission/output \
  --config configs/banana_uav.yaml \
  --settle-seconds 0.5
```

For ROS2 camera topics, `edge/ros2/banana_counter_node.py` keeps the configured
detector in memory and publishes JSON detections on `bananavision/detections`.

## Prediction quality

After inference, audit detections before publishing counts or updating the persistent inventory:

```bash
bananavision prediction-quality runs/mission \
  --output runs/prediction_quality/prediction_quality_report.json \
  --low-confidence 0.45 \
  --high-split-count 3 \
  --max-review-fraction 0.20
```

The report flags low-confidence detections, banana-cluster splits, edge detections, crowded centers, and duplicate-like overlaps. Treat failures as a review gate: inspect crops, tune cluster parameters, or re-run inference before reporting plant counts.

## Plant-coordinate accuracy

When the mission output will be used in GIS, inventory, or navigation software,
validate detected plant coordinates against surveyed or hand-verified GeoJSON
truth:

```bash
bananavision geo-accuracy runs/mission/mission.detections.geojson /data/holdout/plants.truth.geojson \
  --output runs/geo_accuracy/geo_accuracy_report.json \
  --tolerance-m 1 \
  --max-rmse-m 1 \
  --max-p95-m 1.5 \
  --min-recall 0.99
```

The report writes JSON plus CSV match evidence and is required by
`release-audit` for production releases. Use projected meter coordinates when
possible; lon/lat GeoJSON is accepted and measured with haversine distance.

## systemd service

Generate service files on your development machine or directly on the Jetson:

```bash
bananavision deploy-systemd \
  --output edge/systemd \
  --install-dir /opt/bananavision-drone \
  --user bananavision \
  --bin /opt/bananavision-drone/.venv/bin/bananavision \
  --config /opt/bananavision-drone/configs/banana_uav.yaml \
  --detector yolo-seg \
  --model /models/best.pt \
  --watch-dir /data/mission/incoming \
  --mission-output /data/mission/output
```

Copy `bananavision-mission-watch.service` and `bananavision-api.service` to `/etc/systemd/system/`, then enable them with the commands in the generated `README.systemd.md`.
The generated services include conservative hardening (`NoNewPrivileges`,
`PrivateTmp`, `ProtectSystem`, and explicit `ReadWritePaths` for mission data).
`deployment_manifest.json` records the exact preflight/start commands, health
checks, target paths, service names, and operational gates for field handoff.
For the API, `/health` is a liveness probe and `/ready` confirms the config,
detector, runtime fingerprint, model metadata, auth mode, upload limit, and
allowed image suffixes were loaded at service start. Set
`BANANAVISION_API_KEY` in the service environment file to require `X-API-Key` or
`Authorization: Bearer <token>` on `POST /infer`; health checks remain unauthenticated
for local service monitoring.

## Export

```bash
bananavision export weights/best.pt --format onnx --imgsz 1024
bananavision export weights/best.pt --format engine --imgsz 1024 --half --device 0
```

## Release package verification

Before installing a model package on the drone computer, verify the copied ZIP or
folder:

```bash
bananavision release-package-verify /models/banana-v1.zip \
  --output runs/release_package/verify_report.json \
  --require-deployment-artifacts
```

The verifier recalculates every packaged artifact hash and fails if the manifest,
release audit, model, config, evidence files, or deployment manifest were
modified after packaging or missing from the deployable bundle.

Run one known-good image through the installed runtime before production flight:

```bash
bananavision deployment-smoke-test /data/smoke/banana_smoke.jpg \
  --output runs/deployment_smoke/deployment_smoke_report.json \
  --artifacts-dir runs/deployment_smoke/artifacts \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.engine \
  --min-detections 1 \
  --max-image-latency-ms 250
```

This catches target-machine failures that static package checks cannot see:
missing model files, bad CUDA/TensorRT dependencies, unwritable output folders,
unexpected latency, or a detector that returns no banana candidates on the smoke
image.

After running `preflight` and `deployment-smoke-test` on the drone computer,
combine the package and machine evidence:

```bash
bananavision deployment-audit /models/banana-v1 \
  runs/preflight/preflight_report.json \
  edge/systemd/deployment_manifest.json \
  --output runs/deployment/deployment_audit.json \
  --deployment-smoke-report runs/deployment_smoke/deployment_smoke_report.json
```

Do not start production collection until this audit passes.

For routine field operation, the same final gate can be run as one command:

```bash
bananavision drone-ready /models/banana-v1 \
  edge/systemd/deployment_manifest.json \
  /data/smoke/banana_smoke.jpg \
  --output runs/drone_ready \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.engine \
  --min-detections 1 \
  --max-image-latency-ms 250
```

It writes preflight, deployment-smoke, deployment-audit, evidence-manifest, and
summary reports into the output folder.

## Operational checklist

- Run `bananavision preflight`.
- Calibrate GSD per mission.
- Use the same camera orientation used during training.
- Keep flight altitude inside the validated range.
- Log every image, config, model hash, and output.
- Deploy only a `promote-model` manifest that passed locked acceptance, stratified truth coverage, stratified acceptance, and hardware benchmark gates.
- Verify the release package on the deployment machine before installing weights or config.
- Run `bananavision deployment-smoke-test` against a known-good image on the deployment machine.
- Run `bananavision deployment-audit` after installation and before production flight.
- Use `bananavision drone-ready` for the routine one-command final gate.
- Run `bananavision prediction-quality`, then review low-confidence detections and dense clusters.
- Run `bananavision geo-accuracy` before publishing plant coordinates or a release package.

## Mission outputs

Every `bananavision infer` run writes:

- per-image detections;
- `mission.detections.csv`;
- `mission.detections.geojson`;
- `mission.detections.kml`;
- `run_manifest.json`.

Use projected coordinates when possible. If frames overlap and have valid georeferencing, mission output suppresses duplicates within `mission_geo_dedupe_distance_m`.
KML placemarks are emitted only for lon/lat coordinates; projected or pixel-only detections remain in CSV and GeoJSON.

## Persistent inventory

After a mission, update the plant inventory:

```bash
bananavision inventory-update runs/mission/mission.detections.geojson farm_inventory --distance-threshold 1.2
```

This matches detections to existing plants by distance and creates stable IDs for new plants. Use meters for lon/lat or projected coordinates, and pixel distance only for non-georeferenced testing.
Each update also writes a timestamped snapshot under `farm_inventory/snapshots/`, so flights can be compared without losing the prior state.

Compare an earlier snapshot with the latest inventory:

```bash
bananavision inventory-diff \
  farm_inventory/snapshots/inventory_2026-07-03T12-00-00Z.json \
  farm_inventory/inventory.json \
  --output runs/inventory_diff
```

The diff report lists new, missing, and persistent plant IDs. It also writes GeoJSON and KML layers for new and missing plants so agronomy teams can inspect changes in GIS tools or mission software.
