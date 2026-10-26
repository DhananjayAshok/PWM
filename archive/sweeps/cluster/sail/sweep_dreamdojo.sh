#!/usr/bin/env bash
# Orchestrates the DreamDojo sweep (configs/sweeps/dreamdojo.yaml) on Sail from the dev box, without supervision:
# every cycle it asks pwm/wm/dreamdojo/sweep_plan.py what is missing (grid on the main robot -> seed replicates of the
# two best grid points + the best point on the other robots; a rollout evaluation for every finished run) and launches
# it; runs whose job vanished or failed without a STOPPED status file are relaunched (at most 3 tries each). Every 10 min
# it refreshes <results_dir>/sweeps/dreamdojo/<robot>.md. Only `sky jobs queue` / launch calls and status-file reads:
# no compute.
# Restart-safe: rerunning only launches what is missing.
#   setsid nohup bash cluster/sail/sweep_dreamdojo.sh [--gpus 2] [--priority p2] [--cycle 300] \
#       >> <results_dir>/logs/sweep_dreamdojo/orchestrator.log 2>&1 < /dev/null &
set -uo pipefail
source scripts/utils.sh > /dev/null || { echo "Could not source utils"; exit 1; }
gpus=2 priority=p2 cycle=300 report_every=600
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) gpus="$2" ;; --priority) priority="$2" ;; --cycle) cycle="$2" ;;
        *) echo "Unknown flag $1"; exit 1 ;;
    esac
    shift 2
done
log_dir="${results_dir%/}/logs/sweep_dreamdojo"
mkdir -p "$log_dir"
state="$log_dir/state.json"
queue_file="$log_dir/queue.txt"
robots=$(python -c "import yaml; g = yaml.safe_load(open('configs/sweeps/dreamdojo.yaml'))['groups']; print(' '.join([g['main_robot']] + g['confirm_robots']))")
last_report=0
echo "$(date -u +%FT%TZ) orchestrator start (pid $$): $gpus GPUs per run at $priority, robots $robots"
while true; do
    if ! sky jobs queue --skip-finished -a > "$queue_file.tmp" 2>/dev/null; then
        echo "$(date -u +%FT%TZ) sky jobs queue failed; retrying next cycle"
        sleep "$cycle"; continue
    fi
    mv "$queue_file.tmp" "$queue_file"
    while read -r kind robot run tag rest; do
        case "$kind" in
            TRAIN)
                echo "$(date -u +%FT%TZ) launching training $tag"
                bash cluster/sail/launch.sh cluster/sail/train_dreamdojo.yaml "$tag" --priority "$priority" \
                    --gpus "B300:${gpus}" --cpus $((22 * gpus)) --memory $((250 * gpus)) \
                    --env ROBOT="$robot" --env SWEEP_RUN="$run" --env EXPERIMENT_TAG="$tag" --env WANDB_RUN_ID="$tag" \
                    2>&1 | grep -E "Managed Job ID|rror" ;;
            EVAL)
                echo "$(date -u +%FT%TZ) launching rollout eval $tag"
                bash cluster/sail/launch.sh cluster/sail/rollout_eval_dreamdojo.yaml "$tag" --priority "$priority" \
                    --env ROBOT="$robot" --env RUN_TAG="dd-${robot}-nominal-${run}" \
                    2>&1 | grep -E "Managed Job ID|rror" ;;
            INFO)
                echo "$(date -u +%FT%TZ) $robot $run $tag $rest" ;;
        esac
    done < <(python -m pwm.wm.dreamdojo.sweep_plan --active "$queue_file" --state "$state" 2>> "$log_dir/plan_errors.log")
    now=$(date +%s)
    if (( now - last_report >= report_every )); then
        for robot in $robots; do
            python -m pwm.wm.dreamdojo.sweep_report --robot "$robot" > /dev/null 2>> "$log_dir/report_errors.log"
        done
        last_report=$now
    fi
    sleep "$cycle"
done
