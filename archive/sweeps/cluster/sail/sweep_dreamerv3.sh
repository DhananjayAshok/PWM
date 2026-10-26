#!/usr/bin/env bash
# Runs the whole DreamerV3 sweep (configs/sweeps/dreamerv3.yaml) on Sail, unattended, from the dev box. Per robot:
#   1. grid: packs the grid runs k per GPU job (cluster/sail/sweep_dreamerv3.yaml) and keeps a job alive for every
#      pack until all its runs have stopped (a run is done when its summary in <storage_dir>/sweeps/dreamerv3/<robot>/
#      has a stop_reason);
#   2. when all of the robot's grid runs have stopped, picks the best one by the fixed selection metric
#      (pwm/wm/dreamerv3/sweep.py select -> best_grid.json, logged here and in the project log);
#   3. then packs and runs that robot's `seeds` group: groups.seeds_per_run extra seeds of its two best grid runs.
# Robots on hold (<storage_dir>/data/<robot>/nominal/.hold_dreamerv3, set during data maintenance) are skipped.
# Restart-safe: re-running it only launches what is missing. Refreshes <results_dir>/sweeps/dreamerv3/<robot>.md.
#   setsid nohup bash cluster/sail/sweep_dreamerv3.sh [--robots "franka_libero ..."] [--pack 4] [--priority p2] \
#       >> <results_dir>/logs/sweep_dreamerv3/orchestrator.log 2>&1 < /dev/null &
set -uo pipefail
source scripts/utils.sh > /dev/null || { echo "Could not source utils"; exit 1; }
robots="franka_libero gr1_robocasa widowx_bridge google_fractal" pack=4 priority=p2 cycle=600
while [[ $# -gt 0 ]]; do
    case "$1" in
        --robots) robots="$2" ;; --pack) pack="$2" ;; --priority) priority="$2" ;; --cycle) cycle="$2" ;;
        --group) ;;  # accepted for compatibility: all groups are always run in order
        *) echo "Unknown flag $1"; exit 1 ;;
    esac
    shift 2
done
function group_runs() {  # a group's run names (grid: seed 0 only; seeds: each run at its extra seeds)
    python - "$1" <<'PY'
import sys, yaml
g = yaml.safe_load(open("configs/sweeps/dreamerv3.yaml"))["groups"]
name = sys.argv[1]
runs = g[name] if name == "grid" else [f"{r}-seed{s}" for r in g[name] for s in g.get("seeds_per_run", [])]
print(" ".join(runs))
PY
}
function log() { echo "$(date -u +%FT%TZ) $*"; }
function held() { [[ -e "${storage_dir%/}/data/$1/nominal/.hold_dreamerv3" ]]; }
function run_done() { python -m pwm.wm.dreamerv3.sweep done --robot "$1" --run "$2" 2>/dev/null; }

# job_ended <id>: true only if a fresh full queue read shows job <id> in a final state (a failed read counts as not ended)
function job_ended() {
    timeout 300 sky jobs queue -a 2>/dev/null | awk -v i="$1" '$1 == i' \
        | grep -qE " (SUCCEEDED|FAILED[A-Z_]*|CANCELLED) "
}

# keep_group <robot> <group> <active queue>: launch the missing packs of a group; prints the number of unfinished runs
function keep_group() {
    local robot="$1" group="$2" active="$3" open_runs=0
    read -ra runs <<< "$(group_runs "$group")"
    for ((i = 0; i < ${#runs[@]}; i += pack)); do  # packs are fixed by position, so job names are stable
        local members=("${runs[@]:i:pack}") todo=()
        for m in "${members[@]}"; do [[ "$(run_done "$robot" "$m")" == 1 ]] || todo+=("$m"); done
        [[ ${#todo[@]} -eq 0 ]] && continue
        open_runs=$((open_runs + ${#todo[@]}))
        local name="dv3sw-${robot}-${group}-$((i / pack))"
        echo "$active" | grep -q " ${name} " && continue
        # never relaunch a pack within 45 min of its last launch (a queue read that misses it must not duplicate it)
        local stamp="${launch_dir}/${name}"
        [[ -f "$stamp" && $(( $(date +%s) - $(stat -c %Y "$stamp") )) -lt 2700 ]] && continue
        # the stamp holds the id of the pack's last job: relaunch only once that job has ended (a fresh lookup of
        # that id; the shared queue read above has missed live jobs, 2026-10-26 15:00)
        if [[ -s "$stamp" ]] && ! job_ended "$(cat "$stamp")"; then
            log "$name: queue read missed its job $(cat "$stamp"), which has not ended; not relaunching" >&2
            continue
        fi
        touch "$stamp"
        local csv out
        csv=$(IFS=,; echo "${members[*]}")
        log "launching $name: $csv" >&2
        out=$(bash cluster/sail/launch.sh cluster/sail/sweep_dreamerv3.yaml "$name" --env ROBOT="$robot" \
            --env RUNS="$csv" --env EXPERIMENT_TAG="$name" --priority "$priority" 2>&1 | grep -E "Managed Job ID|rror")
        echo "$out" >&2
        grep -oE "Managed Job ID: [0-9]+" <<< "$out" | grep -oE "[0-9]+$" > "$stamp"
    done
    echo "$open_runs"
}

export storage_dir
launch_dir="${results_dir%/}/logs/sweep_dreamerv3/launched"
mkdir -p "$launch_dir"
while true; do
    active=$(timeout 300 sky jobs queue --skip-finished -a 2>/dev/null)
    # a failed or partial read lists no live jobs, which would relaunch every pack: skip the cycle instead
    if ! grep -qE "^Managed jobs|No in-progress managed jobs" <<< "$active" || ! grep -q "In progress tasks\|No in-progress" <<< "$active"; then
        log "could not read the job queue; retrying next cycle"; sleep "$cycle"; continue
    fi
    remaining=0
    for robot in $robots; do
        if held "$robot"; then
            log "$robot: on hold (data/$robot/nominal/.hold_dreamerv3)"; remaining=$((remaining + 1)); continue
        fi
        open=$(keep_group "$robot" grid "$active")
        if [[ "$open" -gt 0 ]]; then remaining=$((remaining + open)); continue; fi
        best_file="${storage_dir%/}/sweeps/dreamerv3/${robot}/best_grid.json"
        if [[ ! -f "$best_file" ]]; then
            chosen=$(python -m pwm.wm.dreamerv3.sweep select --robot "$robot" 2>/dev/null)
            [[ -z "$chosen" ]] && { log "$robot: grid done but no run could be chosen"; remaining=$((remaining + 1)); continue; }
            log "$robot: grid finished; best grid run = $chosen ($(python -c "import json; d=json.load(open('$best_file')); print('openloop_mse64', round(d['value'], 5), 'ranking', d['ranking'])"))"
        fi
        open=$(keep_group "$robot" seeds "$active")
        remaining=$((remaining + open))
    done
    for robot in $robots; do python -m pwm.wm.dreamerv3.sweep_table --robot "$robot" > /dev/null 2>&1; done
    if [[ $remaining -eq 0 ]]; then log "sweep: all groups done on all robots"; break; fi
    sleep "$cycle"
done
