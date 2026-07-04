# Validation Protocol

This is the commercial acceptance gate.

## Metrics

- Count error rate: `abs(predicted_count - true_count) / true_count`.
- Precision and recall using a center-point tolerance based on GSD.
- Duplicate rate: multiple detections assigned to the same plant.
- Missed cluster rate: mats where at least one individual is missed.
- Plant-coordinate RMSE, p95 error, and recall against field truth GeoJSON.
- Edge latency: median and p95 inference time on target hardware.

## Local Cluster Benchmark

Before field validation, run the synthetic grouped-mat benchmark as a regression
smoke test:

```bash
bananavision cluster-benchmark --output runs/cluster_benchmark
```

This generates clustered banana scenes, writes `truth_coverage_report.json`,
`cluster_acceptance_report.json`, and `cluster_benchmark_report.json`, then exits
non-zero if the configured synthetic thresholds fail. It is not field evidence,
but it catches regressions in banana mat splitting before longer training or
release runs.

## Suggested thresholds

For a public beta:

- count error <= 5% on held-out fields;
- recall >= 95%;
- precision >= 95%;
- p95 latency compatible with mission speed and overlap.

For a 1% count-error claim:

- count error <= 1% on multiple farms, dates, altitudes, and cultivars;
- enough annotated plants that the minimum detectable count error is meaningfully below 1%;
- lower confidence bounds for precision and recall are still inside the claimed operating target;
- plant-level geospatial RMSE, p95 error, and recall pass the release limits;
- externally auditable ground truth;
- no tuning on the final acceptance set.

## Validation sample plan

Before collecting the final holdout, calculate the minimum truth support needed
to make the target error rate visible:

```bash
bananavision validation-plan \
  --output runs/validation/validation_plan.json \
  --target-count-error-rate 0.01 \
  --target-cluster-recall-loss 0.01 \
  --target-cluster-full-detection-loss 0.01 \
  --farms 3 \
  --flight-dates 3 \
  --gsd-bands 2 \
  --cultivars 1
```

Use the report's `minimum_support` and `recommended_acceptance_args` values when
setting `truth-coverage`, `stratified-truth-coverage`, `acceptance-batch`, and
`release-audit` thresholds.
This prevents a tiny holdout from making a 1% claim look precise when one missed
plant would already exceed the claimed error.

## CLI gate

Run:

```bash
bananavision acceptance IMAGE.tif TRUTH.json \
  --detector yolo-seg \
  --model weights/best.pt \
  --max-count-error-rate 0.01 \
  --min-precision 0.99 \
  --min-recall 0.99 \
  --min-f1 0.99
```

The command writes `acceptance_report.json` and exits non-zero if the model fails.

For final validation, lock the holdout before batch acceptance:

```bash
bananavision holdout-lock HOLDOUT_IMAGES TRUTH_MANIFEST.json \
  --output runs/holdout/holdout_lock.json \
  --target-count-error-rate 0.01

bananavision holdout-verify runs/holdout/holdout_lock.json \
  --output runs/holdout/holdout_verify.json

bananavision truth-quality TRUTH_MANIFEST.json \
  --images HOLDOUT_IMAGES \
  --output runs/holdout/truth_quality_report.json

bananavision truth-coverage TRUTH_MANIFEST.json \
  --images HOLDOUT_IMAGES \
  --output runs/holdout/truth_coverage_report.json \
  --min-truth-count 1000 \
  --min-cluster-count 100 \
  --min-cluster-truth-count 300 \
  --min-cluster-images 30 \
  --min-cluster-truth-fraction 0.20

bananavision stratified-truth-coverage TRUTH_MANIFEST.json HOLDOUT_METADATA.csv \
  --images HOLDOUT_IMAGES \
  --output runs/stratified_truth_coverage/stratified_truth_coverage_report.json \
  --strata farm --strata flight_date --strata gsd_band --strata cultivar \
  --min-truth-count 50 \
  --min-cluster-count 10 \
  --min-cluster-truth-count 30 \
  --min-cluster-images 5 \
  --min-cluster-truth-fraction 0.20
```

The lock stores image hashes, truth hashes, per-image truth counts, total truth
count, banana-mat cluster support, and the minimum detectable count error and
cluster-loss rates. If any holdout image or annotation changes, `holdout-verify`
exits non-zero.
The truth quality report fails when annotations contain duplicate centers,
singleton group IDs, oversized groups, or out-of-bounds centers. The truth
coverage report fails when the holdout does not contain enough grouped banana
mats to support cluster-splitting claims. The stratified truth-coverage report
fails when that support is missing inside any claimed farm, date, GSD band,
cultivar, or selected field condition.

Build a visual-domain profile from the same locked holdout images:

```bash
bananavision domain-profile HOLDOUT_IMAGES \
  --output runs/domain/domain_profile.json
```

For each mission or release package, compare the target imagery against that
validated profile:

```bash
bananavision domain-check MISSION_IMAGES runs/domain/domain_profile.json \
  --output runs/domain/domain_check_report.json \
  --max-outlier-fraction 0
```

This detects camera, lighting, preprocessing, season, or crop-state shifts before
the release claims are trusted.

Validate plant-level coordinates against surveyed or hand-verified GeoJSON
truth before claiming GIS-ready outputs:

```bash
bananavision geo-accuracy runs/mission/mission.detections.geojson HOLDOUT_PLANTS.truth.geojson \
  --output runs/geo_accuracy/geo_accuracy_report.json \
  --tolerance-m 1 \
  --max-rmse-m 1 \
  --max-p95-m 1.5 \
  --min-recall 0.99
```

This writes `geo_accuracy_report.json` and a CSV of matched plants. It should be
run on the same coordinate system used for inventory or GIS handoff, preferably
projected meters; lon/lat GeoJSON is also supported.

Then run batch acceptance:

```bash
bananavision acceptance-batch HOLDOUT_IMAGES TRUTH_MANIFEST.json \
  --holdout-lock runs/holdout/holdout_lock.json \
  --detector yolo-seg \
  --model weights/best.pt \
  --max-count-error-rate 0.01 \
  --max-mean-image-count-error-rate 0.03 \
  --min-precision 0.99 \
  --min-recall 0.99 \
  --min-f1 0.99 \
  --min-truth-count 1000 \
  --min-cluster-count 100 \
  --min-cluster-recall 0.99 \
  --min-cluster-full-detection-rate 0.99 \
  --min-precision-ci-lower 0.98 \
  --min-recall-ci-lower 0.98
```

This verifies the holdout lock before inference, then writes
`holdout_verify.json` and `acceptance_batch_report.json` with aggregate precision,
recall, F1, total count error, banana-mat cluster split metrics, mean per-image
count error, per-image metrics, sample support, Wilson confidence intervals for
precision/recall and cluster support, and a confidence interval for mean
per-image count error.

Gate the same acceptance report by operating-domain strata:

```bash
bananavision stratified-acceptance runs/acceptance_batch/acceptance_batch_report.json HOLDOUT_METADATA.csv \
  --output runs/stratified_acceptance/stratified_acceptance_report.json \
  --strata farm --strata flight_date --strata gsd_band --strata cultivar \
  --max-count-error-rate 0.01 \
  --min-precision 0.99 \
  --min-recall 0.99 \
  --min-f1 0.99 \
  --min-truth-count 50 \
  --min-cluster-count 10 \
  --min-cluster-recall 0.99 \
  --min-cluster-full-detection-rate 0.99
```

`HOLDOUT_METADATA.csv` must contain `image` and the selected strata columns.
Every stratum should pass before a broad 1% claim is made for multiple farms,
dates, GSD bands, cultivars, or field conditions.

Promote only after acceptance and benchmark gates pass:

```bash
bananavision promote-model weights/best.pt banana-v1 \
  runs/acceptance_batch/acceptance_batch_report.json \
  runs/benchmark/benchmark_report.json \
  --config configs/banana_uav.yaml \
  --max-p95-ms 250
```

`promote-model` refuses failed acceptance reports and benchmark reports missing p95 latency. If `--max-p95-ms` is supplied, it also refuses slow models.

Benchmark the same hardware used on the drone:

```bash
bananavision benchmark IMAGE_OR_DIR --config configs/banana_uav.yaml --runs 5 --warmup 1
```

Calibrate a confidence threshold from a held-out field:

```bash
bananavision calibrate runs/infer/block-a.detections.json TRUTH.json --tolerance-px 24
```

Tune cluster-separation parameters on calibration data:

```bash
bananavision tune-config IMAGE.tif TRUTH.json \
  --config configs/banana_uav.yaml \
  --tolerance-px 24 \
  --max-split-instances 8 --max-split-instances 12 --max-split-instances 16
```

Keep calibration data and final acceptance data separate.

If the mission has a telemetry or capture CSV, audit the actual flight envelope:

```bash
bananavision flight-log-audit /data/mission/flight_log.csv \
  --output runs/flight_log/flight_log_audit.json \
  --config configs/banana_uav.yaml
```

If the log also contains image names and capture positions, verify frame
coverage:

```bash
bananavision capture-coverage /data/mission/capture_log.csv \
  --images /data/mission/images \
  --output runs/capture_coverage/capture_coverage_report.json \
  --require-image-files \
  --min-images 200 \
  --max-position-gap-m 35
```

Before handing off counts or inventory updates, run the mission delivery audit:

```bash
bananavision mission-audit \
  runs/infer/run_manifest.json \
  runs/mission_quality/mission_quality_report.json \
  runs/prediction_quality/prediction_quality_report.json \
  --flight-check-report runs/flight_check/flight_check_report.json \
  --flight-log-report runs/flight_log/flight_log_audit.json \
  --capture-coverage-report runs/capture_coverage/capture_coverage_report.json \
  --require-capture-coverage \
  --domain-check-report runs/domain/domain_check_report.json \
  --geo-accuracy-report runs/geo_accuracy/geo_accuracy_report.json \
  --require-geo-accuracy \
  --output runs/mission_audit/mission_audit.json
```

Generate an audit artifact for stakeholders:

```bash
bananavision field-report \
  --run-manifest runs/infer/run_manifest.json \
  --mission-audit-report runs/mission_audit/mission_audit.json \
  --mission-quality-report runs/mission_quality/mission_quality_report.json \
  --prediction-quality-report runs/prediction_quality/prediction_quality_report.json \
  --flight-check-report runs/flight_check/flight_check_report.json \
  --flight-log-report runs/flight_log/flight_log_audit.json \
  --capture-coverage-report runs/capture_coverage/capture_coverage_report.json \
  --domain-check-report runs/domain/domain_check_report.json \
  --geo-accuracy-report runs/geo_accuracy/geo_accuracy_report.json \
  --validation-plan-report runs/validation/validation_plan.json \
  --truth-quality-report runs/holdout/truth_quality_report.json \
  --truth-coverage-report runs/holdout/truth_coverage_report.json \
  --stratified-truth-coverage-report runs/stratified_truth_coverage/stratified_truth_coverage_report.json \
  --acceptance-report runs/acceptance/acceptance_report.json \
  --stratified-acceptance-report runs/stratified_acceptance/stratified_acceptance_report.json \
  --benchmark-report runs/benchmark/benchmark_report.json \
  --tuning-report runs/tuning/tuning_report.json \
  --cluster-review-report runs/cluster_review/block-a.cluster_review.json \
  --release-audit-report runs/release_audit/release_audit.json \
  --model-manifest models/registry/latest.json
```

Generate the model card from the same evidence:

```bash
bananavision model-card \
  --output docs/MODEL_CARD.generated.md \
  --model-name "BananaVision field model" \
  --version banana-v1 \
  --model-manifest models/registry/latest.json \
  --acceptance-report runs/acceptance_batch/acceptance_batch_report.json \
  --benchmark-report runs/benchmark/benchmark_report.json \
  --mission-quality-report runs/mission_quality/mission_quality_report.json \
  --prediction-quality-report runs/prediction_quality/prediction_quality_report.json \
  --flight-log-report runs/flight_log/flight_log_audit.json \
  --domain-check-report runs/domain/domain_check_report.json \
  --geo-accuracy-report runs/geo_accuracy/geo_accuracy_report.json \
  --validation-plan-report runs/validation/validation_plan.json \
  --stratified-acceptance-report runs/stratified_acceptance/stratified_acceptance_report.json \
  --truth-quality-report runs/holdout/truth_quality_report.json \
  --truth-coverage-report runs/holdout/truth_coverage_report.json \
  --stratified-truth-coverage-report runs/stratified_truth_coverage/stratified_truth_coverage_report.json
```

The generated card states whether the provided acceptance evidence supports,
fails, or does not prove a 1% claim, whether locked truth annotations passed
quality checks, whether every claimed field-condition stratum has truth support
and passed acceptance, and whether grouped banana-mat truth coverage exists.

When a grouped-mat gate fails, generate a cluster-review report before changing
labels, split parameters, or model weights:

```bash
bananavision cluster-review runs/infer/block-a.detections.json TRUTH_MANIFEST.json \
  --output runs/cluster_review/block-a.cluster_review.json \
  --tolerance-px 24 \
  --crops-dir runs/cluster_review/crops
```

Keep this report with calibration evidence so reviewers can see which banana
mats caused under-split or over-split failures.

Run a final release audit:

```bash
bananavision release-audit \
  --acceptance-report runs/acceptance_batch/acceptance_batch_report.json \
  --stratified-acceptance-report runs/stratified_acceptance/stratified_acceptance_report.json \
  --benchmark-report runs/benchmark/benchmark_report.json \
  --mission-quality-report runs/mission_quality/mission_quality_report.json \
  --prediction-quality-report runs/prediction_quality/prediction_quality_report.json \
  --holdout-verify-report runs/acceptance_batch/holdout_verify.json \
  --validation-plan-report runs/validation/validation_plan.json \
  --truth-quality-report runs/holdout/truth_quality_report.json \
  --truth-coverage-report runs/holdout/truth_coverage_report.json \
  --stratified-truth-coverage-report runs/stratified_truth_coverage/stratified_truth_coverage_report.json \
  --flight-check-report runs/flight_check/flight_check_report.json \
  --flight-log-report runs/flight_log/flight_log_audit.json \
  --domain-check-report runs/domain/domain_check_report.json \
  --geo-accuracy-report runs/geo_accuracy/geo_accuracy_report.json \
  --model-manifest models/registry/latest.json \
  --model-card docs/MODEL_CARD.generated.md \
  --field-report runs/reports/field_report.html \
  --max-count-error-rate 0.01 \
  --min-truth-count 1000 \
  --min-cluster-count 100 \
  --min-cluster-truth-count 300 \
  --min-cluster-images 30 \
  --min-cluster-truth-fraction 0.20 \
  --min-cluster-recall 0.99 \
  --min-cluster-full-detection-rate 0.99 \
  --min-precision-ci-lower 0.98 \
  --min-recall-ci-lower 0.98 \
  --max-p95-ms 250 \
  --max-geo-rmse-m 1 \
  --max-geo-p95-m 1.5 \
  --min-geo-recall 0.99
```

This is the final publication gate. It fails if acceptance, stratified truth coverage or stratified acceptance for 1% claims, validation-plan support for a 1% claim, truth-quality evidence, sample support, CI lower bounds, benchmark latency, QA reports, holdout verification, flight-envelope evidence, supplied post-flight log evidence, domain-shift evidence, plant-coordinate accuracy, model manifest, model card, or field report are missing or outside configured limits.

Build the evidence manifest for handoff traceability:

```bash
bananavision evidence-manifest \
  --output runs/evidence/evidence_manifest.json \
  --run-manifest runs/infer/run_manifest.json \
  --mission-audit-report runs/mission_audit/mission_audit.json \
  --mission-quality-report runs/mission_quality/mission_quality_report.json \
  --prediction-quality-report runs/prediction_quality/prediction_quality_report.json \
  --flight-check-report runs/flight_check/flight_check_report.json \
  --flight-log-report runs/flight_log/flight_log_audit.json \
  --capture-coverage-report runs/capture_coverage/capture_coverage_report.json \
  --domain-check-report runs/domain/domain_check_report.json \
  --geo-accuracy-report runs/geo_accuracy/geo_accuracy_report.json \
  --validation-plan-report runs/validation/validation_plan.json \
  --truth-quality-report runs/holdout/truth_quality_report.json \
  --truth-coverage-report runs/holdout/truth_coverage_report.json \
  --stratified-truth-coverage-report runs/stratified_truth_coverage/stratified_truth_coverage_report.json \
  --acceptance-report runs/acceptance_batch/acceptance_batch_report.json \
  --stratified-acceptance-report runs/stratified_acceptance/stratified_acceptance_report.json \
  --benchmark-report runs/benchmark/benchmark_report.json \
  --cluster-review-report runs/cluster_review/block-a.cluster_review.json \
  --model-manifest models/registry/latest.json \
  --model-card docs/MODEL_CARD.generated.md \
  --field-report runs/reports/field_report.html \
  --release-audit-report runs/release_audit/release_audit.json \
  --model weights/best.engine \
  --config configs/banana_uav.yaml \
  --require release_audit_report \
  --require acceptance_report \
  --require benchmark_report \
  --require validation_plan_report \
  --require truth_quality_report \
  --require truth_coverage_report \
  --require stratified_truth_coverage_report \
  --require model_manifest \
  --require model_card \
  --require field_report \
  --require model \
  --require config
```

This index makes every release artifact reviewable by label, path, SHA256 hash,
reported status, and required/missing state.

Package only after the final audit passes:

```bash
bananavision release-package runs/release_audit/release_audit.json \
  --output dist \
  --package-name banana-v1 \
  --model weights/best.engine \
  --config configs/banana_uav.yaml \
  --model-manifest models/registry/latest.json \
  --model-card docs/MODEL_CARD.generated.md \
  --field-report runs/reports/field_report.html \
  --evidence-manifest runs/evidence/evidence_manifest.json \
  --mission-audit-report runs/mission_audit/mission_audit.json \
  --acceptance-report runs/acceptance_batch/acceptance_batch_report.json \
  --stratified-acceptance-report runs/stratified_acceptance/stratified_acceptance_report.json \
  --benchmark-report runs/benchmark/benchmark_report.json \
  --mission-quality-report runs/mission_quality/mission_quality_report.json \
  --prediction-quality-report runs/prediction_quality/prediction_quality_report.json \
  --holdout-verify-report runs/acceptance_batch/holdout_verify.json \
  --validation-plan-report runs/validation/validation_plan.json \
  --truth-quality-report runs/holdout/truth_quality_report.json \
  --truth-coverage-report runs/holdout/truth_coverage_report.json \
  --stratified-truth-coverage-report runs/stratified_truth_coverage/stratified_truth_coverage_report.json \
  --flight-check-report runs/flight_check/flight_check_report.json \
  --flight-log-report runs/flight_log/flight_log_audit.json \
  --domain-check-report runs/domain/domain_check_report.json \
  --geo-accuracy-report runs/geo_accuracy/geo_accuracy_report.json \
  --deployment-manifest edge/systemd/deployment_manifest.json
```

The release package contains a manifest with SHA256 hashes for every artifact and
refuses failed release audits by default. Existing package folders are not reused;
pass `--overwrite` only when intentionally regenerating the same package name.

Verify the package before publishing or deploying it:

```bash
bananavision release-package-verify dist/banana-v1.zip \
  --output runs/release_package/verify_report.json \
  --require-deployment-artifacts
```

This recomputes artifact hashes, validates the manifest hash, checks release
audit status, and fails if a production package has been changed after build or
lacks deployment-critical model, config, evidence, or service manifest artifacts.

After installing on the target drone computer, run a target-machine inference
smoke test:

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

Then run the final deployment audit:

```bash
bananavision deployment-audit dist/banana-v1 \
  runs/preflight/preflight_report.json \
  edge/systemd/deployment_manifest.json \
  --output runs/deployment/deployment_audit.json \
  --deployment-smoke-report runs/deployment_smoke/deployment_smoke_report.json
```

This combines strict release-package verification, the target machine's preflight
report, the generated deployment manifest, and a real inference smoke report. It
should pass before any production flight.

For final handoff, run `evidence-manifest` again with
`--deployment-smoke-report` and `--deployment-audit-report` so the installed
runtime evidence is indexed alongside the release artifacts.

For routine target-machine checks, use the combined command:

```bash
bananavision drone-ready dist/banana-v1 \
  edge/systemd/deployment_manifest.json \
  /data/smoke/banana_smoke.jpg \
  --output runs/drone_ready \
  --config configs/banana_uav.yaml \
  --detector yolo-seg \
  --model weights/best.engine \
  --min-detections 1 \
  --max-image-latency-ms 250
```

This writes the preflight, smoke, deployment-audit, evidence-manifest, and
summary reports from a single field command.

Before publishing the repository, run:

```bash
bananavision publication-audit . \
  --output runs/publication/publication_audit.json
```

This verifies public-facing repo files, CI, security policy, contribution guide,
operator docs, and validation/release-flow documentation.

## Why this matters

Software can be ready today, but an error-rate claim is a measured property of a trained model on a defined operating domain.
