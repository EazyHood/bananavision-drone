# Failure Modes

BananaVision should fail loudly when evidence is missing or the mission is
outside the validated operating domain. Treat failures as protection against bad
counts, not as inconveniences to bypass.

## Preflight Fails

Common causes:

- model path missing;
- output directory not writable;
- too little disk space;
- optional ML dependencies not installed;
- no georeference evidence when required.

Action: fix the environment or model path, then rerun `bananavision preflight`.

## Flight Check Fails

Common causes:

- planned GSD differs from the validated config;
- front or side overlap is too low;
- speed/exposure combination produces excessive motion blur;
- GSD cannot be measured or estimated.

Action: adjust altitude, camera settings, overlap, speed, or exposure. Do not
reuse a model outside its validated GSD band without a new acceptance run.

## Flight Log Audit Fails

Common causes:

- actual GSD drifted outside the validated range during flight;
- overlap dropped below the required mission envelope;
- speed/exposure caused excessive computed motion blur;
- telemetry or capture CSV is missing required GSD/camera-geometry fields.

Action: inspect the failed rows in the CSV report, remove invalid captures or
refly affected areas, and do not use failed rows for production counts or release
evidence.

## Capture Coverage Fails

Common causes:

- capture log references images that are missing on disk;
- capture positions are missing from the CSV;
- large jumps between consecutive capture positions suggest dropped frames or uncovered strips;
- repeated positions suggest duplicate captures rather than new coverage.

Action: sync or recover missing images, inspect the flight/camera controller log,
remove invalid captures or refly the affected strip, then rerun
`bananavision capture-coverage`.

## Domain Check Fails

Common causes:

- new camera or lens;
- different image processing pipeline;
- extreme shadows, glare, haze, or overcast conditions not represented in holdout;
- different crop age, cultivar, disease state, or ground cover;
- seasonal or farm-management changes.

Action: inspect the flagged images, add representative annotated data, retrain or
tune as needed, then rebuild the holdout/domain profile and rerun acceptance.

## Mission Quality Fails

Common causes:

- blurry frames;
- underexposure or overexposure;
- resolution below the validated minimum;
- missing georeferencing.

Action: refly affected areas or remove failed captures from production metrics.
Do not treat bad capture as model failure.

## Prediction Quality Fails

Common causes:

- low confidence detections;
- dense banana mats split into too many or too few plants;
- detections near image edges;
- duplicate-like overlapping detections;
- zero detections in an image where plants are expected.

Action: review crops, tune splitting thresholds on calibration data, or retrain
with corrected annotations. Keep final acceptance data separate.

## Mission Audit Fails

Common causes:

- run manifest has no images or too few detections for the expected mission;
- mission-quality or prediction-quality failed;
- required flight-check or domain-check evidence is missing;
- supplied flight-log, capture-coverage, or geo-accuracy evidence failed;
- runtime fingerprint is missing from the mission manifest.

Action: do not publish counts or update inventory from this mission. Fix the
failing artifact, review flagged detections/captures, refly affected areas if
needed, and rerun `bananavision mission-audit`.

## Geo Accuracy Fails

Common causes:

- detections are not tied to an orthomosaic, world file, GeoTIFF transform, or equivalent geometry;
- plant centers are shifted because GSD or camera geometry is wrong;
- duplicate suppression merged or kept the wrong overlapping-frame detections;
- field-truth GeoJSON uses a different CRS or datum;
- too many plants are unmatched against surveyed or hand-verified truth.

Action: verify the coordinate system, regenerate georeferenced outputs, inspect
the CSV match report, tune duplicate suppression or splitting thresholds, and
rerun `bananavision geo-accuracy`. Do not publish plant coordinates if this gate
fails.

## Truth Quality Fails

Common causes:

- duplicate or extremely close plant centers;
- a `group_id` appears on only one plant because of a typo;
- a grouped banana mat has more individuals than the configured maximum;
- a center is outside the image bounds.

Action: correct the annotation file, rerun `bananavision truth-quality`, then
rerun `truth-coverage`, `holdout-lock`, and acceptance on the corrected locked
holdout. Do not use failed truth-quality evidence for a 1% claim.

## Acceptance Fails

Common causes:

- count error above threshold;
- precision, recall, or F1 below threshold;
- confidence intervals too weak;
- too few truth plants to prove a 1% claim.
- one farm, date, GSD band, cultivar, or other metadata stratum fails even when the global average passes.

Action: add field data, improve labels, retrain, and rerun on a locked holdout.
Do not lower thresholds to make a production claim pass.

## Stratified Truth Coverage Fails

Common causes:

- metadata CSV is missing images from the locked holdout;
- one field condition has too few annotated plants;
- one field condition has too few grouped banana mats;
- metadata strata do not match the farms, dates, GSD bands, or cultivars claimed in the validation plan.

Action: collect or annotate more holdout data for the failed condition, fix
metadata, rerun `bananavision stratified-truth-coverage`, then rerun acceptance
and `release-audit`.

## Release Audit Fails

Common causes:

- missing holdout verification;
- failed or weak acceptance report;
- missing or failed stratified truth-coverage evidence for a 1% claim;
- missing or failed stratified acceptance evidence for a 1% claim;
- failed or missing truth-quality evidence for a 1% claim;
- missing flight-check or domain-check evidence;
- supplied flight-log audit evidence failed;
- missing or failing geo-accuracy evidence;
- benchmark latency too high;
- QA reports warning/failing;
- missing model card, field report, or promoted model manifest.

Action: fix the failing gate and rerun `bananavision release-audit`. A failed
audit can be packaged only with `--allow-failed-audit`, and that package is
marked exploratory.

## Release Package Fails

Common causes:

- release audit status is not `pass`;
- required artifact path is missing;
- package folder or ZIP already exists.

Action: provide the missing artifact, pass the release audit, or use `--overwrite`
only when intentionally regenerating the same package name.

## Release Package Verify Fails

Common causes:

- an artifact was edited after packaging;
- the ZIP or folder copy is incomplete;
- the manifest hash does not match;
- the package is exploratory but is being treated as production;
- release audit evidence inside the package did not pass.
- `--require-deployment-artifacts` found missing model, config, evidence, or deployment manifest files.

Action: do not deploy the package. Rebuild it from passing release evidence and
rerun `bananavision release-package-verify` on the exact folder or ZIP copied to
the drone computer.

## Deployment Audit Fails

Common causes:

- release-package verification failed on the installed bundle;
- `preflight` failed or warned and warnings were not explicitly allowed;
- deployment smoke report is missing or failed;
- smoke inference did not process images, returned too few detections, exceeded latency, or did not write mission outputs;
- runtime fingerprint is missing from the preflight report;
- generated systemd manifest is missing preflight, mission-watch, or `/ready` checks;
- deployment service files listed in the manifest are incomplete.

Action: do not fly. Fix the installed package, rerun `bananavision preflight` on
the drone computer, rerun `bananavision deployment-smoke-test` on a known-good
image, regenerate systemd artifacts if needed, then rerun `bananavision
deployment-audit` or the combined `bananavision drone-ready` command.
