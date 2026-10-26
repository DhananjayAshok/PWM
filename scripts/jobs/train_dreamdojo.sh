#!/usr/bin/env bash
# Fine-tune DreamDojo 2B on one robot's data with native actions until its validation loss stops improving
# (pwm/wm/dreamdojo/train.py), one process per GPU on one node. Restart-safe: rerunning resumes from the newest
# checkpoint; a finished run (<run_dir>/STOPPED) exits at once.
#   bash scripts/jobs/train_dreamdojo.sh --robot <robot> [--dataset nominal] [--run_name <name>] [--run_dir <dir>]
#       [--gpus all] [--max_steps 0]
# Needs <model_dir>/dreamdojo (scripts/jobs/prepare_dreamdojo.sh). Cluster wrapper: cluster/sail/train_dreamdojo.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict TRAIN_DREAMDOJO_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array TRAIN_DREAMDOJO_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

run_name="${ARGS["run_name"]}"
[[ "$run_name" == "none" ]] && run_name="dd-${ARGS["robot"]}-${ARGS["dataset"]}"
run_dir="${ARGS["run_dir"]}"
[[ "$run_dir" == "none" ]] && run_dir="${storage_dir%/}/checkpoints/${run_name}"
gpus="${ARGS["gpus"]}"
[[ "$gpus" == "all" ]] && gpus=$(nvidia-smi -L | wc -l)
mkdir -p "$run_dir"
echo "[train_dreamdojo] ${ARGS["robot"]} on $gpus GPU(s) -> $run_dir"
( use_env dreamdojo
  export EXPERIMENT_TAG="${EXPERIMENT_TAG:-$run_name}" WANDB_RUN_ID="${WANDB_RUN_ID:-$run_name}"
  # single node: rendezvous on localhost (a recovered job's node list can be stale)
  python -m torch.distributed.run --nproc_per_node "$gpus" --nnodes 1 --master_addr 127.0.0.1 \
      --master_port "${ARGS["port"]}" -m pwm.wm.dreamdojo.train --robot "${ARGS["robot"]}" \
      --dataset "${storage_dir%/}/data/${ARGS["robot"]}/${ARGS["dataset"]}" --run_dir "$run_dir" \
      --init "${ARGS["init"]}" --max_steps "${ARGS["max_steps"]}" \
      --val_every_clips "${ARGS["val_every_clips"]}" \
      --save_every "${ARGS["save_every"]}" 2>&1 | tee -a "$run_dir/train.log" )
