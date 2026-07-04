# Model Card

Generate model cards from release evidence instead of editing this file by hand:

```bash
bananavision model-card \
  --output docs/MODEL_CARD.generated.md \
  --model-name "BananaVision field model" \
  --version banana-field-v1 \
  --model-manifest models/registry/latest.json \
  --acceptance-report runs/acceptance_batch/acceptance_batch_report.json \
  --benchmark-report runs/benchmark/benchmark_report.json \
  --mission-quality-report runs/mission_quality/mission_quality_report.json \
  --prediction-quality-report runs/prediction_quality/prediction_quality_report.json \
  --flight-log-report runs/flight_log/flight_log_audit.json \
  --domain-check-report runs/domain/domain_check_report.json \
  --geo-accuracy-report runs/geo_accuracy/geo_accuracy_report.json \
  --dataset-quality-report runs/quality/quality_report.json \
  --validation-plan-report runs/validation/validation_plan.json \
  --stratified-acceptance-report runs/stratified_acceptance/stratified_acceptance_report.json \
  --truth-quality-report runs/holdout/truth_quality_report.json \
  --truth-coverage-report runs/holdout/truth_coverage_report.json \
  --stratified-truth-coverage-report runs/stratified_truth_coverage/stratified_truth_coverage_report.json
```

The generated card includes:

- model and manifest hashes;
- intended use;
- 1% claim status;
- acceptance metrics;
- validation-plan support for 1% claims;
- stratified truth coverage by farm, date, GSD band, cultivar, or selected field condition;
- stratified acceptance by farm, date, GSD band, cultivar, or selected field condition;
- truth-quality status for locked annotations;
- sample support and confidence intervals;
- grouped banana-mat truth coverage;
- mission, prediction, domain, and geospatial quality evidence;
- edge latency;
- known limitations.

Do not publish a 1% error claim from a manually edited model card. The claim must come from a locked acceptance report, validation plan, passing stratified truth coverage, passing stratified acceptance, and passing truth-quality evidence with enough annotated plants and statistical support.
