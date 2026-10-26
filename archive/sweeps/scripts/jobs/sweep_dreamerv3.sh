#!/usr/bin/env bash
# Train several DreamerV3 sweep runs (configs/sweeps/dreamerv3.yaml) side by side on one GPU. The port's RSSM is a
# Python loop over time steps, so one run is bound by kernel-launch latency and leaves the GPU mostly idle; packing k
# runs per GPU multiplies throughput. Each run is restart-safe on its own (pwm/wm/dreamerv3/train.py); finished runs
# exit at once, so rerunning the same command resumes whatever is unfinished. GPU-hours are booked 1/k per run.
#   bash scripts/jobs/sweep_dreamerv3.sh --robot <robot> --runs "baseline,lr3e-4,..." [--dataset nominal]
#       [--ckpt_root <dir>] [--workers_per_run 4] [--overrides key=value,...]
# Cluster wrapper: cluster/sail/sweep_dreamerv3.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SWEEP_DREAMERV3_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SWEEP_DREAMERV3_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

robot="${ARGS["robot"]}"
root="${ARGS["ckpt_root"]}"
[[ "$root" == "none" ]] && root="${storage_dir%/}/checkpoints"
IFS=',' read -ra runs <<< "${ARGS["runs"]}"
log_dir="${results_dir%/}/logs/sweep_dreamerv3/${robot}"
mkdir -p "$log_dir"
export PWM_GPU_SHARE=$(python -c "print(1 / ${#runs[@]})")
export PWM_CACHE_WORKERS=$(( ${OMP_NUM_THREADS:-8} ))
export OMP_NUM_THREADS=$(( ${OMP_NUM_THREADS:-8} / ${#runs[@]} > 0 ? ${OMP_NUM_THREADS:-8} / ${#runs[@]} : 1 ))
extra=(--override "loader_workers=${ARGS["workers_per_run"]}")
if [[ "${ARGS["overrides"]}" != "none" ]]; then
    IFS=',' read -ra items <<< "${ARGS["overrides"]}"
    for item in "${items[@]}"; do extra+=(--override "$item"); done
fi
pids=()
for run in "${runs[@]}"; do
    name="dv3-${robot}-${run}"   # the nominal sweep; other datasets carry their name
    [[ "${ARGS["dataset"]}" != "nominal" ]] && name="dv3-${robot}-${ARGS["dataset"]}-${run}"
    ( use_env dreamerv3
      exec python -m pwm.wm.dreamerv3.train --robot "$robot" --dataset "${ARGS["dataset"]}" --sweep_run "$run" \
          --run_name "$name" --ckpt_dir "${root}/${name}" "${extra[@]}" ) \
        >> "${log_dir}/${name}.log" 2>&1 &
    pids+=($!)
    echo "[sweep] $name started (pid $!), log ${log_dir}/${name}.log"
done
status=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "[sweep] ${runs[$i]} exited with an error"; status=1; }
done
python -m pwm.wm.dreamerv3.sweep_table --robot "$robot" --dataset "${ARGS["dataset"]}" || true
exit $status
