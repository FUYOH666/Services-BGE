#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHD_DIR="$PROJECT_DIR/launchd"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/Services-BGE"

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
  install     Copy plist files to LaunchAgents and create log directory
  uninstall   Stop services and remove plist files
  start       Start both services via launchd
  stop        Stop both services via launchd
  restart     Stop then start both services
  status      Show PID, health, and log paths
  logs        Tail logs from both services (Ctrl+C to stop)

EOF
    exit 1
}

cmd_install() {
    echo "=== Installing BGE Services ==="

    mkdir -p "$AGENTS_DIR" "$LOG_DIR"

    for svc in "${SERVICES[@]}"; do
        src="$LAUNCHD_DIR/$svc.plist"
        dst="$AGENTS_DIR/$svc.plist"

        if [ ! -f "$src" ]; then
            echo "ERROR: $src not found"
            exit 1
        fi

        cp "$src" "$dst"
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

        # Kill anything on the port first
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "  Killing stale process on port $port..."
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 1
        fi

        # Unload if already loaded, ignore errors
        launchctl bootout "gui/$(id -u)/$svc" 2>/dev/null || true

        launchctl bootstrap "gui/$(id -u)" "$plist"
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

        if launchctl print "gui/$(id -u)/$svc" &>/dev/null; then
            launchctl bootout "gui/$(id -u)/$svc"
            echo "  Stopped $name"
        else
            echo "  $name not running"
        fi
    done

    # Clean up any lingering processes on the ports
    for port in "${PORTS[@]}"; do
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
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

        # Check if loaded in launchd
        if launchctl print "gui/$(id -u)/$svc" &>/dev/null; then
            pid=$(launchctl print "gui/$(id -u)/$svc" 2>/dev/null | grep -m1 "pid =" | awk '{print $3}' || echo "?")
            printf "launchd=loaded  pid=%-8s " "$pid"
        else
            printf "launchd=unloaded               "
        fi

        # Check health endpoint
        health=$(curl -s --max-time 2 "http://localhost:$port/healthz" 2>/dev/null || echo "")
        if [ -n "$health" ]; then
            ready=$(echo "$health" | python3 -c "import json,sys; print(json.load(sys.stdin).get('ready','?'))" 2>/dev/null || echo "?")
            device=$(echo "$health" | python3 -c "import json,sys; print(json.load(sys.stdin).get('device','?'))" 2>/dev/null || echo "?")
            printf "health=ok  ready=%s  device=%s" "$ready" "$device"
        else
            printf "health=unreachable"
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

# --- Main ---
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
    *)         usage ;;
esac
