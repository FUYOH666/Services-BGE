# Contributing

Thanks for helping. Bug reports, model results and pull requests are all welcome.

## Reporting

- **Bugs**: open an [issue](https://github.com/FUYOH666/Services-BGE/issues/new/choose) with the request you sent, the response or log lines (`~/Library/Logs/Services-BGE/*.err`), your Mac model and macOS version.
- **Model results**: if you evaluated a model on your own data with `eval/run_eval.py`, share the `compare.py` table (no private data needed). It helps everyone choose.
- **Security issues**: privately, see [SECURITY.md](SECURITY.md).

## Development setup

Apple Silicon Mac, Python 3.12, [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/FUYOH666/Services-BGE.git && cd Services-BGE
uv sync --extra dev
uv run --extra dev pytest -q     # fake models, no downloads, < 1 s
```

Run your branch without touching services you already have on 9001/9002:

```bash
EMBEDDING_PORT=9101 ./scripts/run_service.sh embedding &
RERANKER_PORT=9102 ./scripts/run_service.sh reranker &
uv run python scripts/parity_check.py      # compare with 9001/9002
uv run python scripts/test_services.py --embed-url http://127.0.0.1:9101 --rerank-url http://127.0.0.1:9102
```

## Pull requests

- Keep the default API responses backward compatible; add new behaviour behind a request field or setting. The contract tests in `tests/test_service_endpoints.py` must pass.
- Add or update tests for behaviour changes. Tests must not download models.
- New model: add a profile in `profiles/`, document it in `docs/MODELS.md`, and include eval numbers in the PR.
- Update `CHANGELOG.md` and the relevant doc in `docs/`.
- No absolute personal paths, IP addresses or secrets in tracked files.

Coding agents: read [AGENTS.md](AGENTS.md) first.
