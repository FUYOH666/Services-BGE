#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHD_DIR="$PROJECT_DIR/launchd"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/Services-BGE"
LAUNCHD_DOMAIN="gui/$(id -u)"

SERVICES=(
    "com.services-bge.embedding"
    "com.services-bge.reranker"
)

PORTS=(9001 9002)
NAMES=(embedding reranker)

usage() {
    cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  install     Render plist files into LaunchAgents and create log directory
  uninstall   Stop services and remove plist files
  start       Start both services via launchd
  stop        Stop both services via launchd
  restart     Stop then start both services
  status      Show PID, readiness, and log paths
  logs        Tail logs from both services (Ctrl+C to stop)
  rotate-logs Archive and truncate log files larger than ROTATE_MAX_MB (default 10),
              keeping ROTATE_KEEP (default 5) gzip archives per file

EOF
    exit 1
}

port_pids() {
    local port="$1"
    lsof -ti :"$port" 2>/dev/null || true
}

wait_for_port_free() {
    local port="$1"
    local attempts="${2:-20}"
    local pids=""

    for ((i = 0; i < attempts; i++)); do
        pids="$(port_pids "$port")"
        if [ -z "$pids" ]; then
            return 0
        fi
        sleep 0.5
    done

    echo "$pids"
    return 1
}

launchd_path() {
    local uv_bin
    uv_bin="$(command -v uv 2>/dev/null || true)"
    if [ -n "$uv_bin" ]; then
        printf '%s:%s\n' "$(dirname "$uv_bin")" "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    else
        printf '%s\n' "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    fi
}

render_plist() {
    local src="$1"
    local dst="$2"
    local resolved_path
    resolved_path="$(launchd_path)"

    python3 - "$src" "$dst" "$PROJECT_DIR" "$HOME" "$resolved_path" "$LOG_DIR" <<'PY'
from pathlib import Path
import sys

src, dst, project_dir, home_dir, launchd_path, log_dir = sys.argv[1:]
text = Path(src).read_text()
text = text.replace("__PROJECT_DIR__", project_dir)
text = text.replace("__HOME__", home_dir)
text = text.replace("__LAUNCHD_PATH__", launchd_path)
text = text.replace("__LOG_DIR__", log_dir)
Path(dst).write_text(text)
PY
}

cmd_install() {
    echo "=== Installing BGE Services ==="

    mkdir -p "$AGENTS_DIR" "$LOG_DIR"
    chmod +x "$PROJECT_DIR/scripts/run_service.sh"

    for svc in "${SERVICES[@]}"; do
        src="$LAUNCHD_DIR/$svc.plist"
        dst="$AGENTS_DIR/$svc.plist"

        if [ ! -f "$src" ]; then
            echo "ERROR: $src not found"
            exit 1
        fi

        render_plist "$src" "$dst"
        echo "  Installed $dst"
    done

    echo ""
    echo "Done. Run './scripts/service.sh start' to start services."
}

cmd_uninstall() {
    echo "=== Uninstalling BGE Services ==="
    cmd_stop 2>/dev/null || true

    for svc in "${SERVICES[@]}"; do
        dst="$AGENTS_DIR/$svc.plist"
        if [ -f "$dst" ]; then
            rm -f "$dst"
            echo "  Removed $dst"
        fi
    done

    echo "Done. Log directory preserved at $LOG_DIR"
}

cmd_start() {
    echo "=== Starting BGE Services ==="

    for i in "${!SERVICES[@]}"; do
        svc="${SERVICES[$i]}"
        port="${PORTS[$i]}"
        name="${NAMES[$i]}"
        plist="$AGENTS_DIR/$svc.plist"

        if [ ! -f "$plist" ]; then
            echo "ERROR: $plist not found. Run './scripts/service.sh install' first."
            exit 1
        fi

        launchctl bootout "$LAUNCHD_DOMAIN/$svc" 2>/dev/null || true

        if pids="$(wait_for_port_free "$port" 12)"; then
            :
        else
            echo "ERROR: Port $port is already in use by PID(s): $pids"
            echo "Refusing to kill unrelated processes automatically."
            exit 1
        fi

        launchctl bootstrap "$LAUNCHD_DOMAIN" "$plist"
        echo "  Started $name (port $port)"
    done

    echo ""
    echo "Models loading... check status in ~10 seconds:"
    echo "  ./scripts/service.sh status"
}

cmd_stop() {
    echo "=== Stopping BGE Services ==="

    for i in "${!SERVICES[@]}"; do
        svc="${SERVICES[$i]}"
        name="${NAMES[$i]}"
        port="${PORTS[$i]}"

        if launchctl print "$LAUNCHD_DOMAIN/$svc" &>/dev/null; then
            launchctl bootout "$LAUNCHD_DOMAIN/$svc"
            echo "  Stopped $name"
        else
            echo "  $name not running"
        fi

        if pids="$(wait_for_port_free "$port" 12)"; then
            :
        elif [ -n "$pids" ]; then
            echo "  Warning: port $port is still in use by PID(s): $pids"
            echo "  Leaving that process untouched."
        fi
    done

    echo "Done."
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

cmd_status() {
    echo "=== BGE Services Status ==="
    echo ""

    for i in "${!SERVICES[@]}"; do
        svc="${SERVICES[$i]}"
        port="${PORTS[$i]}"
        name="${NAMES[$i]}"

        printf "%-12s " "[$name]"

        if launchctl print "$LAUNCHD_DOMAIN/$svc" &>/dev/null; then
            pid=$(launchctl print "$LAUNCHD_DOMAIN/$svc" 2>/dev/null | awk '/pid =/{print $3; exit}' || echo "?")
            printf "launchd=loaded  pid=%-8s " "$pid"
        else
            printf "launchd=unloaded               "
        fi

        health="$(curl -s --max-time 2 "http://localhost:$port/readyz" 2>/dev/null || true)"
        if [ -n "$health" ]; then
            ready=$(echo "$health" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ready','?'))" 2>/dev/null || echo "?")
            device=$(echo "$health" | python3 -c "import json,sys; print(json.load(sys.stdin).get('device','?'))" 2>/dev/null || echo "?")
            status=$(echo "$health" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "?")
            printf "ready=%s  status=%s  device=%s" "$ready" "$status" "$device"
        else
            printf "ready=unreachable"
        fi
        echo ""
    done

    echo ""
    echo "Logs:"
    echo "  $LOG_DIR/embedding.log"
    echo "  $LOG_DIR/reranker.log"
}

cmd_logs() {
    if [ ! -d "$LOG_DIR" ]; then
        echo "No log directory at $LOG_DIR"
        exit 1
    fi

    echo "=== Tailing BGE logs (Ctrl+C to stop) ==="
    tail -f "$LOG_DIR"/embedding.log "$LOG_DIR"/reranker.log "$LOG_DIR"/embedding.err "$LOG_DIR"/reranker.err 2>/dev/null
}

cmd_rotate_logs() {
    local max_mb="${ROTATE_MAX_MB:-10}"
    local keep="${ROTATE_KEEP:-5}"

    if [ ! -d "$LOG_DIR" ]; then
        echo "No log directory at $LOG_DIR"
        exit 0
    fi

    echo "=== Rotating BGE logs larger than ${max_mb} MB (keep $keep) ==="
    local stamp
    stamp="$(date +%Y%m%d-%H%M%S)"

    for f in "$LOG_DIR"/*.log "$LOG_DIR"/*.err; do
        [ -f "$f" ] || continue
        local size_mb
        size_mb=$(( $(stat -f %z "$f") / 1024 / 1024 ))
        if [ "$size_mb" -lt "$max_mb" ]; then
            continue
        fi

        # launchd keeps the file open in append mode, so copy + truncate is safe.
        gzip -c "$f" > "$f.$stamp.gz"
        : > "$f"
        echo "  Rotated $(basename "$f") (${size_mb} MB) -> $(basename "$f").$stamp.gz"

        # Keep only the newest $keep archives for this file.
        # shellcheck disable=SC2012
        ls -1t "$f".*.gz 2>/dev/null | tail -n +"$((keep + 1))" | while read -r old; do
            rm -f "$old"
        done
    done
    echo "Done."
}

if [ $# -lt 1 ]; then
    usage
fi

case "$1" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    logs)      cmd_logs ;;
    rotate-logs) cmd_rotate_logs ;;
    *)         usage ;;
esac
