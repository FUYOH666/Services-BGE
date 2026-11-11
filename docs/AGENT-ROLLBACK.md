# Agent memory: disconnect and roll back

Connecting an agent to [agent memory](AGENT-MEMORY.md) is meant to be a **safe, additive bonus**: each agent gets exactly one new MCP server entry named `services-bge`, and nothing else in its configuration, roles or prompts changes. This page shows how to undo it, agent by agent, and how to delete the stored memory.

- [What the setup adds](#what-the-setup-adds)
- [Disconnect one agent](#disconnect-one-agent): [Claude Code](#claude-code) · [Cursor](#cursor) · [OpenClaw](#openclaw) · [Hermes](#hermes)
- [Restore a backup instead](#restore-a-backup-instead)
- [Delete the stored memory](#delete-the-stored-memory)
- [What rollback does not touch](#what-rollback-does-not-touch)
- [Check that it is gone](#check-that-it-is-gone)

## What the setup adds

| Agent | Config file | Added | Backup made before the change |
|---|---|---|---|
| Claude Code | `~/.claude.json` (user scope) | MCP server `services-bge` → `scripts/mcp.sh` | `~/.claude.json.bak-services-bge-<date>` |
| Cursor | `~/.cursor/mcp.json` | key `services-bge` inside `mcpServers` | `~/.cursor/mcp.json.bak-services-bge-<date>` |
| OpenClaw | `~/.openclaw/openclaw.json` | entry `services-bge` in `mcp.servers` (timeout 600 s) | `~/.openclaw/openclaw.json.bak-services-bge-<date>` |
| Hermes | `~/.hermes/config.yaml` | block `services-bge` under `mcp_servers:` (timeout 600 s) | `~/.hermes/config.yaml.bak-services-bge-<date>` |

The server runs over stdio: each agent starts `scripts/mcp.sh` itself when it needs memory and stops it afterwards. There is no extra background service, launchd agent or open port to remove.

## Disconnect one agent

Removing the entry is enough; the rest of the agent's configuration stays as it was.

### Claude Code

```bash
claude mcp remove services-bge --scope user
claude mcp list                      # services-bge is no longer listed
```

### Cursor

Open `~/.cursor/mcp.json` and delete the `"services-bge": { … }` entry inside `mcpServers` (keep the comma rules of JSON: no trailing comma). Or with one command:

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path.home() / ".cursor" / "mcp.json"
data = json.loads(p.read_text())
data.get("mcpServers", {}).pop("services-bge", None)
p.write_text(json.dumps(data, indent=2) + "\n")
print("remaining:", list(data.get("mcpServers", {})))
PY
```

Then reload Cursor (or toggle the server off in Cursor Settings → MCP).

### OpenClaw

```bash
openclaw mcp unset services-bge
openclaw mcp reload
openclaw mcp list                    # services-bge is no longer listed
```

OpenClaw's own memory embedder (its openai-compatible provider pointing at `http://127.0.0.1:9001/v1`) is a different setting and keeps working.

### Hermes

```bash
hermes mcp remove services-bge
hermes mcp list
```

Or delete the `services-bge:` block under `mcp_servers:` in `~/.hermes/config.yaml`. If Hermes runs as a gateway, restart it so the change is picked up.

### Memory auto-sync

If you turned on `./scripts/service.sh memory-sync on`, turn it off with `./scripts/service.sh memory-sync off`. That removes the launchd agent `com.services-bge.memory-sync`; the index keeps its last state.

## Restore a backup instead

Every config file was copied before the change. Restoring the copy brings the file back exactly as it was at that moment — but it also discards any other change you made to that file **after** the setup, so prefer the commands above.

```bash
ls ~/.claude.json.bak-services-bge-* ~/.cursor/mcp.json.bak-services-bge-* \
   ~/.openclaw/openclaw.json.bak-services-bge-* ~/.hermes/config.yaml.bak-services-bge-*

# example: restore Cursor's file
cp ~/.cursor/mcp.json.bak-services-bge-<date> ~/.cursor/mcp.json
```

To see what the setup changed, diff a file against its backup, e.g. `diff ~/.cursor/mcp.json.bak-services-bge-<date> ~/.cursor/mcp.json`.

## Delete the stored memory

Disconnecting an agent keeps what it remembered. To delete memory too:

- **From an agent**, while it is still connected: `forget(collection="memory", everything=true)` (per collection; `list_collections` shows them).
- **Directly in Qdrant**: memory collections are the ones prefixed `mcp_`.

```bash
curl -s http://127.0.0.1:6333/collections | python3 -c "import json,sys; print([c['name'] for c in json.load(sys.stdin)['result']['collections'] if c['name'].startswith('mcp_')])"
curl -s -X DELETE http://127.0.0.1:6333/collections/mcp_memory     # one collection, permanent
```

Collections without the `mcp_` prefix belong to other projects and are never touched by the MCP server; do not delete them as part of this rollback.

## What rollback does not touch

- The embedding and reranker services on `9001` / `9002` and every project that calls them.
- OpenClaw's memory embedder setting and any other agent setting outside the `services-bge` entry.
- Qdrant collections without the `mcp_` prefix.

## Check that it is gone

```bash
claude mcp list | grep services-bge || echo "Claude Code: not connected"
grep -q '"services-bge"' ~/.cursor/mcp.json && echo "Cursor: still there" || echo "Cursor: not connected"
openclaw mcp list | grep services-bge || echo "OpenClaw: not connected"
hermes mcp list | grep services-bge || echo "Hermes: not connected"
```
