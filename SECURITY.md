# Security

## Supported versions

Security fixes go to the latest release on `main`.

## Reporting a vulnerability

Please report privately first, not in a public issue:

- Email: [iamfuyoh@gmail.com](mailto:iamfuyoh@gmail.com)
- Telegram: [@ScanovichAI](https://t.me/ScanovichAI)
- or GitHub [private vulnerability reporting](https://github.com/FUYOH666/Services-BGE/security/advisories/new)

You will get an answer within a few days. Please include steps to reproduce and the version or commit.

## Deployment notes

- The services bind to `127.0.0.1` by default and have no authentication unless `SERVICE_API_KEY` is set. Before binding to `0.0.0.0` (LAN, Tailscale), set a key.
- Request limits (`*_MAX_ITEMS`, `*_MAX_DOCUMENTS`, `*_MAX_TEXT_CHARS`, `*_MAX_LENGTH_CAP`) protect the GPU from oversized requests; keep them on for shared deployments.
- `*_TRUST_REMOTE_CODE=true` executes code from the model repository. Enable it only for models whose code you have reviewed.
