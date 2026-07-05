# Model Card: BananaVision Real Aerial v2.0

- Generated at: 2026-07-05T03:36:53+00:00
- Version: real-v2.0
- Architecture: YOLOv8m detection (real aerial banana, per-plant labels)
- Model SHA256: d8104c0a30ac270b957eead79683c7ee73641c4f4fef8dd219fa6aa8e361e9b1
- Manifest SHA256: f406af84b838883f1e8c3814bdae575dbf91401c60cf14e9c002dd018ff77357

## Intended Use

Banana plant instance detection and counting from UAV imagery inside the validated operating domain.

## Claim Status

- Status: **not accepted**
- Detail: Acceptance report did not pass.

## Validation Evidence

- Validation plan: not provided.
- Passed: None
- Images: 50
- Truth count: 1166
- Prediction count: 1030
- Count error rate: 0.1166
- Precision: 0.8960
- Recall: 0.7920
- F1: 0.8408
- Annotated banana clusters: 0
- Cluster truth plants: 0
- Cluster recall: unknown
- Fully detected cluster rate: unknown
- Precision Wilson CI: unknown
- Recall Wilson CI: unknown
- Minimum detectable count error rate: unknown
- Stratified acceptance: not provided.

## Operational QA Evidence

- Mission quality: not provided.
- Prediction quality: not provided.
- Flight log audit: not provided.
- Domain check: not provided.
- Geo accuracy: not provided.
- Dataset quality: not provided.
- Truth quality: not provided.
- Truth coverage: not provided.
- Stratified truth coverage: not provided.

## Edge Performance

- Benchmark report: not provided.

## Known Limitations

- Dense banana mats with severe leaf overlap can still need human review.
- Severe shadows, motion blur, low overlap, or off-domain flight altitude can reduce reliability.
- Off-domain camera, lighting, processing, season, or crop-state shifts require domain-check review.
- EXIF GPS geotags are capture-level evidence, not plant-level coordinates; plant coordinates require a passing geo-accuracy report.
- Do not claim a 1% error rate without a locked, representative, annotated holdout set.

## Notes

Trained on real aerial UAV banana imagery (count-banana-plants, CC-BY-4.0). Held-out test: recall 0.79, precision 0.90, mAP50 0.90. Single region; validate/fine-tune for another farm.
