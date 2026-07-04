# Security Policy

## Reporting

Please report security issues privately to the project maintainers before public
disclosure. If this repository is mirrored under an organization, use that
organization's private vulnerability reporting channel.

## Sensitive data

Do not publish:

- private farm boundaries or exact customer locations;
- raw drone imagery without data-owner approval;
- API keys, cloud credentials, SSH keys, or autopilot credentials;
- logs that expose operator identities or precise flight locations.

## Deployment safety

BananaVision is decision-support software. It should not control flight-critical
autopilot behavior. Keep drone navigation, obstacle avoidance, and emergency
procedures under the certified flight stack and trained operator.

Before field deployment, run:

```bash
bananavision preflight
bananavision flight-check
bananavision domain-check
bananavision release-audit
```

Production packages should be distributed through `bananavision release-package`
so hashes and release evidence can be checked.

## API exposure

Do not expose the inference API directly to the public internet. Bind to the
drone or field network only, place it behind a gateway if remote access is
needed, and set `BANANAVISION_API_KEY` in the service environment so
`POST /infer` requires `X-API-Key` or a bearer token. Keep the default upload limit
unless the validated camera workflow requires larger images.
