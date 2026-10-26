#!/usr/bin/env bash
# Start (or restart) every dev-box loop the unattended run needs, e.g. after the dev box is rebuilt: the reaper (the
# sweep orchestrators were archived on 2026-10-26). Skips any that is already running. Run from the project root once
# `sky api info` works.
#   bash cluster/sail/resume_unattended.sh
source configs/config.env || { echo "configs/config.env not found"; exit 1; }
# a rebuilt dev box loses ~/.bashrc; the loops below need HF_HOME (the Sail volume's Hugging Face cache)
export HF_HOME="${HF_HOME:-/persist/cache/huggingface}"
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }
sky jobs queue > /dev/null 2>&1 || { echo "sky cannot reach the API server; reconnect it first"; exit 1; }
logs="${results_dir%/}/logs"
mkdir -p "$logs"

function start() {   # start <pattern> <log> <command...>
    local pattern="$1" log="$2"; shift 2
    if ps -eo cmd | grep -Eq "$pattern"; then echo "[resume] already running: $pattern"; return; fi
    setsid nohup "$@" >> "$log" 2>&1 < /dev/null &
    disown
    echo "[resume] started: $*"
}
start "reaper\.sh" "$logs/reaper.log" bash cluster/sail/reaper.sh
