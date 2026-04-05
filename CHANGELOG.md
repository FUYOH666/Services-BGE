# Changelog

## 2026-04-05

- aligned `launchd`, `.env`, and service startup through `scripts/run_service.sh`
- made `service.sh`, `start_all.sh`, and `stop_all.sh` refuse to kill unrelated processes on ports `9001` and `9002`
- added explicit `livez` and `readyz` endpoints while keeping `healthz` as a readiness alias
- kept `/health` as a backward-compatible readiness alias for older clients
- moved blocking model inference off the event loop and serialized inference with an async lock
- normalized tokenizer parallelism to the canonical `false` string across runtime config and launchd
- added lightweight pytest coverage for readiness and happy-path API responses using fake model stubs
- updated README and smoke test commands for the new operational behavior
