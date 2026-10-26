#!/usr/bin/env bash
# Train DreamerV3's world model on one robot's data until its validation loss stops improving (pwm/wm/dreamerv3/train.py).
# Restart-safe: rerunning resumes from the newest checkpoint; a finished run (STOPPED.json) exits at once.
#   bash scripts/jobs/train_dreamerv3.sh --robot <robot> [--dataset nominal] [--run_name <name>] [--ckpt_dir <dir>]
#       [--overrides key=value,key=value]
# Cluster wrapper: cluster/sail/train_dreamerv3.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict TRAIN_DREAMERV3_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array TRAIN_DREAMERV3_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

run_name="${ARGS["run_name"]}"
[[ "$run_name" == "none" ]] && run_name="dv3-${ARGS["robot"]}-${ARGS["dataset"]}"
ckpt_dir="${ARGS["ckpt_dir"]}"
[[ "$ckpt_dir" == "none" ]] && ckpt_dir="${storage_dir%/}/checkpoints/${run_name}"
override_flags=()
if [[ "${ARGS["overrides"]}" != "none" ]]; then
    IFS=',' read -ra items <<< "${ARGS["overrides"]}"
    for item in "${items[@]}"; do override_flags+=(--override "$item"); done
fi
( use_env dreamerv3
  exec python -m pwm.wm.dreamerv3.train --robot "${ARGS["robot"]}" --dataset "${ARGS["dataset"]}" \
      --run_name "$run_name" --ckpt_dir "$ckpt_dir" "${override_flags[@]}" )
