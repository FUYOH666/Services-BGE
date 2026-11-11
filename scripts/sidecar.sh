#!/bin/bash
# Run model profiles (profiles/*.env) as side services next to the production ones.
# Sidecars are for experiments: they run in the background, are not started at
# login, and never touch the launchd services on 9001/9002.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROFILES_DIR="$PROJECT_DIR/profiles"
LOG_DIR="$HOME/Library/Logs/Services-BGE"
RUN_DIR="${TMPDIR:-/tmp}/services-bge-sidecars"
READY_TIMEOUT="${SIDECAR_READY_TIMEOUT:-900}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [profile...]

Commands:
  list                  Profiles with service, port, backend, model and state
  up <profile>...       Start profiles in the background and wait until ready
  down <profile>...|all Stop running sidecars
  status                Running sidecars and their readiness
  logs <profile>        Tail a sidecar log

The first start of a profile downloads its model from Hugging Face.
Logs: $LOG_DIR/sidecar-<profile>.log
EOF
    exit 1
}

profile_value() {
    # profile_value <profile> <KEY> -> value from the profile file (quotes stripped)
    local file="$PROFILES_DIR/$1.env"
    sed -n "s/^$2=//p" "$file" | tail -1 | sed -e 's/^"//' -e 's/"$//'
}

profile_port() {
    local service
    service="$(profile_value "$1" PROFILE_SERVICE)"
    if [ "$service" = "embedding" ]; then
        profile_value "$1" EMBEDDING_PORT
    else
        profile_value "$1" RERANKER_PORT
    fi
}

require_profile() {
    if [ ! -f "$PROFILES_DIR/$1.env" ]; then
        echo "Unknown profile '$1'. Available:"
        ls "$PROFILES_DIR"/*.env | xargs -n1 basename | sed 's/\.env$/  /'
        exit 1
    fi
}

listener_pid() {
    lsof -ti tcp:"$1" -sTCP:LISTEN 2>/dev/null | head -1 || true
}

is_ready() {
    curl -sf --max-time 2 "http://127.0.0.1:$1/readyz" >/dev/null 2>&1
}

cmd_list() {
    printf "%-26s %-10s %-6s %-22s %-38s %s\n" PROFILE SERVICE PORT BACKEND MODEL STATE
    for file in "$PROFILES_DIR"/*.env; do
        local name service port backend model state pid
        name="$(basename "$file" .env)"
        service="$(profile_value "$name" PROFILE_SERVICE)"
        port="$(profile_port "$name")"
        if [ "$service" = "embedding" ]; then
            backend="$(profile_value "$name" EMBEDDING_BACKEND)"
            model="$(profile_value "$name" EMBEDDING_MODEL)"
        else
            backend="$(profile_value "$name" RERANKER_BACKEND)"
            model="$(profile_value "$name" RERANKER_MODEL)"
        fi
        pid="$(listener_pid "$port")"
        if [ -z "$pid" ]; then
            state="stopped"
        elif is_ready "$port"; then
            state="ready"
        else
            state="loading"
        fi
        printf "%-26s %-10s %-6s %-22s %-38s %s\n" "$name" "$service" "$port" "$backend" "$model" "$state"
    done
}

cmd_up() {
    [ $# -ge 1 ] || usage
    for name in "$@"; do
        require_profile "$name"
        local port pid log
        port="$(profile_port "$name")"
        log="$LOG_DIR/sidecar-$name.log"
        pid="$(listener_pid "$port")"
        if [ -n "$pid" ]; then
            if is_ready "$port"; then
                echo "[$name] already serving on :$port (pid $pid)"
                continue
            fi
            echo "[$name] port $port is busy (pid $pid); refusing to start"
            exit 1
        fi

        echo "[$name] starting on :$port (log: $log)"
        nohup "$PROJECT_DIR/scripts/run_service.sh" --profile "$name" >>"$log" 2>&1 &
        echo $! >"$RUN_DIR/$name.pid"
    done

    for name in "$@"; do
        local port waited=0
        port="$(profile_port "$name")"
        printf "[%s] waiting for readiness" "$name"
        until is_ready "$port"; do
            if ! kill -0 "$(cat "$RUN_DIR/$name.pid" 2>/dev/null || echo 0)" 2>/dev/null && [ -z "$(listener_pid "$port")" ]; then
                echo " failed. Last log lines:"
                tail -20 "$LOG_DIR/sidecar-$name.log"
                exit 1
            fi
            if [ "$waited" -ge "$READY_TIMEOUT" ]; then
                echo " timed out after ${READY_TIMEOUT}s (still downloading or loading?)"
                exit 1
            fi
            sleep 3
            waited=$((waited + 3))
            printf "."
        done
        echo " ready: http://127.0.0.1:$port"
    done
}

stop_one() {
    local name="$1" port pidfile pid listener
    port="$(profile_port "$name")"
    pidfile="$RUN_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        pid="$(cat "$pidfile")"
        kill "$pid" 2>/dev/null || true
        rm -f "$pidfile"
    fi
    for _ in $(seq 1 20); do
        [ -z "$(listener_pid "$port")" ] && break
        sleep 0.5
    done
    listener="$(listener_pid "$port")"
    if [ -n "$listener" ]; then
        # Only stop a leftover listener that belongs to this project.
        if ps -o command= -p "$listener" | grep -q "$PROJECT_DIR"; then
            kill "$listener" 2>/dev/null || true
        else
            echo "[$name] port $port is held by an unrelated process (pid $listener); leaving it"
            return
        fi
    fi
    echo "[$name] stopped"
}

cmd_down() {
    [ $# -ge 1 ] || usage
    if [ "$1" = "all" ]; then
        for pidfile in "$RUN_DIR"/*.pid; do
            [ -f "$pidfile" ] || continue
            stop_one "$(basename "$pidfile" .pid)"
        done
        return
    fi
    for name in "$@"; do
        require_profile "$name"
        stop_one "$name"
    done
}

cmd_status() {
    local any=0
    for pidfile in "$RUN_DIR"/*.pid; do
        [ -f "$pidfile" ] || continue
        any=1
        local name port
        name="$(basename "$pidfile" .pid)"
        port="$(profile_port "$name")"
        if is_ready "$port"; then
            echo "[$name] ready on :$port"
        elif [ -n "$(listener_pid "$port")" ]; then
            echo "[$name] loading on :$port"
        else
            echo "[$name] not running (stale pid file)"
        fi
    done
    [ "$any" = 1 ] || echo "No sidecars started from this machine. Try: $(basename "$0") list"
}

cmd_logs() {
    [ $# -eq 1 ] || usage
    require_profile "$1"
    tail -f "$LOG_DIR/sidecar-$1.log"
}

[ $# -ge 1 ] || usage
command="$1"
shift
case "$command" in
    list)   cmd_list ;;
    up)     cmd_up "$@" ;;
    down)   cmd_down "$@" ;;
    status) cmd_status ;;
    logs)   cmd_logs "$@" ;;
    *)      usage ;;
esac
