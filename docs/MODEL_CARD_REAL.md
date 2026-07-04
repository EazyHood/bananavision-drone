# Model Card: BananaVision Real UAV v1.0

- Generated at: 2026-07-04T15:43:51+00:00
- Version: real-v1.0
- Architecture: YOLOv8s detection (14k real UAV banana tiles)
- Model SHA256: 794492aa2a0259b66203cce973df5fb052496f1cd99229d4275a46f00f5e6bc2
- Manifest SHA256: 378ac58a6280ae7c14f1c313fb583f5aeb83bf376024f925cf38836261379777

## Intended Use

Banana plant instance detection and counting from UAV imagery inside the validated operating domain.

## Claim Status

- Status: **not accepted**
- Detail: Acceptance report did not pass.

## Validation Evidence

- Validation plan: not provided.
- Passed: None
- Images: 4611
- Truth count: 6017
- Prediction count: 5839
- Count error rate: 0.0296
- Precision: 0.4720
- Recall: 0.4580
- F1: 0.4649
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

Entrenado con imágenes UAV REALES (DS-v1 AI-BananaMapping, Zenodo 20945958, CC-BY-4.0). Dataset de baja altitud (escala de planta). Valida en tu finca antes de confiar en el conteo.
