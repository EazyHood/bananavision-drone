# Commercial Readiness

## Ready now

- Python package.
- CLI.
- API server.
- Preflight readiness report.
- Flight-envelope check for GSD, overlap, and motion blur.
- Post-flight telemetry/capture log audit for actual GSD, overlap, speed, and exposure drift.
- Capture coverage audit for missing images, duplicate positions, and large capture gaps.
- Mission image quality audit.
- Visual domain-shift check against validated holdout imagery.
- One-command mission processing workflow.
- Mission-watch runtime for companion computers.
- Prediction quality gate for uncertain or crowded detections.
- Mission delivery audit gate before reporting counts or updating inventory.
- Hardened systemd units and a deployment manifest for Linux/Jetson operation.
- Baseline detector.
- Train/validate/export commands.
- Geospatial outputs.
- Plant-level geospatial accuracy audit against truth GeoJSON.
- KML output for lon/lat mission detections.
- EXIF GPS geotag detection for drone capture QA.
- Persistent plant inventory with stable IDs, snapshots, and flight-to-flight diffs.
- Dataset audit.
- Acceptance gate with strict threshold failure.
- Validation sample-size planner for 1% count-error and grouped-mat claims.
- Holdout locking and verification by image/truth hashes.
- Truth quality gate for annotation defects before release evidence is trusted.
- Truth coverage gate for grouped banana-mat support in validation data.
- Stratified truth-coverage gate for per-condition holdout support.
- Batch acceptance gate for locked holdout folders with confidence intervals and sample-support gates.
- Stratified acceptance gate by farm, flight date, GSD band, cultivar, or other field condition.
- Banana mat/group separation metrics and gates for cluster recall and full-cluster detection.
- Latency benchmark report.
- Run manifests with config/model hashes.
- Mission-level duplicate suppression for overlapping frames.
- COCO/LabelMe annotation conversion.
- Group-aware dataset splitting.
- Dataset tiling with polygon clipping.
- Dataset quality gate.
- Active-learning review queue.
- Review crop export.
- Threshold calibration report.
- Cluster-separation tuning report and tuned config export.
- Cluster-review report and crops for grouped banana mat under/over-split failures.
- Model registry manifest.
- Strict model promotion gate.
- Evidence-backed model card generator.
- Release audit gate for publication readiness.
- Evidence manifest with artifact hashes, reported pass/warn/fail states, and required-release checks.
- Release package builder with hashed artifacts and ZIP export.
- Release package verifier for folder, manifest, ZIP integrity, and deployment-artifact completeness.
- Deployment smoke test that runs real inference on the target drone computer.
- Deployment audit gate for installed package, target-machine preflight, and service manifest readiness.
- One-command drone-ready gate for final preflight, smoke inference, deployment audit, and evidence indexing.
- Publication audit for GitHub readiness, CI, security, contribution, and operator docs.
- Static HTML field report.
- Tests and CI.

## Needed before selling or claiming 1% error

- Field-specific annotated dataset.
- Validation plan matching the claimed farms, dates, GSD bands, and cultivars.
- Model card with dataset coverage.
- Hardware benchmark on the actual drone computer.
- Acceptance report from a locked validation set.
- Passing stratified acceptance report for every claimed farm, date, GSD band, cultivar, or field condition.
- Passing truth-quality report for the locked validation annotations.
- Cluster-annotated holdout coverage for grouped banana mats.
- Passing truth-coverage report for grouped banana-mat sample support.
- Passing stratified truth-coverage report for each claimed farm, date, GSD band, cultivar, or field condition.
- Cluster-review evidence for any grouped-mat failures used during calibration or remediation.
- Passing flight-envelope report from the target operating profile.
- Passing capture-coverage report from target mission imagery when capture positions are available.
- Passing domain-shift report from target mission imagery.
- Passing plant-coordinate accuracy report from field-truth GeoJSON.
- Passing evidence manifest that includes every release-critical artifact.
- Passing deployment smoke test on the actual drone computer.
- Passing drone-ready report from the actual drone computer.
- Versioned model registry.
  The repo includes manifest registration and strict promotion gates; a real product should store promoted weights in durable object storage or release artifacts.

## Product principle

The code should stay open. The value is in transparent tooling, field data quality, and reproducible validation, not in hiding basic agricultural computer vision behind an expensive black box.
