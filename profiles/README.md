# Model profiles

A profile is a small env file that describes one model deployment: which service,
which backend, which model, which port. Profiles let you run challengers next to the
production services and compare them with `eval/run_eval.py`, without editing `.env`.

```bash
./scripts/sidecar.sh list                       # profiles, ports, running state
./scripts/sidecar.sh up qwen3-reranker-0.6b     # start on its own port, wait until ready
./scripts/sidecar.sh status
./scripts/sidecar.sh down qwen3-reranker-0.6b   # or: down all
```

Precedence: variables on the command line > profile > `.env` > built-in defaults.
Every profile sets all model-specific keys (including empty ones) so nothing leaks in
from `.env`.

Ports: `9001`/`9002` are the production control. Sidecars use `9011+`
(odd = embedding, even = reranker) so they never collide with it.

See [docs/MODELS.md](../docs/MODELS.md) for the backend reference, verification status
and how to add a profile.
