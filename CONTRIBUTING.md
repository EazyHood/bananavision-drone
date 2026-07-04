# Contributing

BananaVision is intended to stay open and auditable. Contributions should improve
field reliability, validation transparency, or drone deployment safety.

## Local checks

Run these before opening a pull request:

```bash
python -m ruff check .
python -m pytest -q
```

## Data and model claims

- Do not commit private farm data, customer imagery, API keys, or drone logs with
  identifiable locations unless the owner has explicitly approved publication.
- Do not claim a 1% count-error rate without a locked holdout, passing
  `acceptance-batch`, `flight-check`, `domain-check`, `benchmark`, `model-card`,
  and `release-audit`.
- Keep calibration/tuning images separate from final acceptance images.

## Pull request checklist

- Tests cover the behavior or gate being changed.
- Documentation explains any new operator command or validation requirement.
- Release gates fail honestly when evidence is missing or weak.
- Generated artifacts under `runs/` are examples only, not production evidence.
