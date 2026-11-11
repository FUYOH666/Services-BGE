# Operations

How the services run on a Mac, how to change them safely, and what to check when something looks wrong.

## Service management

```bash
./scripts/service.sh install      # render launchd plists into ~/Library/LaunchAgents
./scripts/service.sh start        # bootstrap both agents (RunAtLoad + KeepAlive)
./scripts/service.sh status       # launchd state, PID, readiness
./scripts/service.sh restart      # after editing .env or pulling a new version
./scripts/service.sh logs         # tail all four log files
./scripts/service.sh memory-sync on   # keep the agents' shared memory index fresh (every 30 min)
./scripts/service.sh rotate-logs  # gzip + truncate files over ROTATE_MAX_MB (10), keep ROTATE_KEEP (5)
./scripts/service.sh stop
./scripts/service.sh uninstall    # stop and remove the plists; logs are kept
```

- launchd labels: `com.services-bge.embedding`, `com.services-bge.reranker`.
- Each agent runs `scripts/run_service.sh <service>`, which loads `.env`, decides on offline mode and `exec`s uvicorn through `uv run`.
- `ThrottleInterval` is 10 s, so a crash loop restarts at most every 10 s.
- `service.sh` and `start_all.sh` refuse to kill an unrelated process that holds 9001/9002.

## Logs

`~/Library/Logs/Services-BGE/`:

| File | Content |
|---|---|
| `embedding.err`, `reranker.err` | Application log (startup, warnings, slow requests, errors) |
| `embedding.log`, `reranker.log` | uvicorn access log (`*_ACCESS_LOG=false` turns it off) |
| `sidecar-<profile>.log` | Sidecars started with `scripts/sidecar.sh` |

Per-request inference lines are `DEBUG`; requests over `*_SLOW_REQUEST_MS` are `WARNING`. To keep the folder small, run `rotate-logs` from time to time (or from cron).

## Deploying a change safely

1. Run the tests: `uv run --extra dev pytest`.
2. Start the new code on side ports next to production:
   ```bash
   EMBEDDING_PORT=9101 ./scripts/run_service.sh embedding &
   RERANKER_PORT=9102 ./scripts/run_service.sh reranker &
   ```
3. Compare with production: `uv run python scripts/parity_check.py` (defaults: A = 9001/9002, B = 9101/9102). Same model and precision must give `PARITY: PASS`; for a deliberate change (fp16, library upgrade) read the drift numbers.
4. Smoke-test: `uv run python scripts/test_services.py --embed-url http://127.0.0.1:9101 --rerank-url http://127.0.0.1:9102`.
5. Stop the side copies, merge, `./scripts/service.sh restart`, and run `scripts/test_services.py` once more.

## Offline start and pinned models

When `HF_HUB_OFFLINE` is unset, `run_service.sh` checks the local Hugging Face cache and turns offline mode on if the model is there, so a boot without network does not stall on `huggingface.co` retries. The startup line shows `hf_offline=1`. Pin a version with `*_MODEL_REVISION=<commit sha>`; the snapshot is resolved locally when cached.

## Memory

- Production footprint is ~3.7 GB per service in fp32 (`footprint -p <pid>`; `ps` RSS under-reports MPS memory).
- `*_MPS_CACHE_THRESHOLD_MB=0` (default) empties the MPS allocator cache after every request; this keeps memory flat at a small latency cost.
- fp16 saves 0.6–1.2 GB per service; measured trade-offs are in [ROADMAP.md](ROADMAP.md#fp16-on-the-current-stack-m4-max-torch-213-measured-2026-11-11-with-parity_checkpy).

## Troubleshooting

| Symptom | Check |
|---|---|
| `status` shows `ready=unreachable` | `tail ~/Library/Logs/Services-BGE/*.err`; first start downloads models and can take minutes |
| `Port 9001 is already in use` | `lsof -i :9001` — another process holds the port; the scripts will not kill it |
| `413` from a client | the client sends more texts or longer texts than the limits; raise `*_MAX_ITEMS` / `*_MAX_TEXT_CHARS` or batch on the client |
| `401` after enabling a key | the client must send `Authorization: Bearer <key>` or `X-API-Key` |
| A sidecar never becomes ready | `./scripts/sidecar.sh logs <profile>`; model id, `trust_remote_code`, or not enough memory |
| `resource_tracker: leaked semaphore objects` on shutdown | known macOS/PyTorch teardown warning; harmless unless startup fails |
