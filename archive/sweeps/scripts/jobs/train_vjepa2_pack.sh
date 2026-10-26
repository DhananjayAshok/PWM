#!/usr/bin/env bash
# Several V-JEPA 2-AC sweep runs of one robot packed onto one GPU (pwm/wm/vjepa2/train.py, one process each, world 1):
# builds the frame cache once, then trains the runs side by side. Each run is restart-safe on its own (own checkpoint
# directory); finished runs exit at once, so rerunning the pack resumes only what is left.
#   bash scripts/jobs/train_vjepa2_pack.sh --robot <robot> --runs "lr_1e-4 lr_4e-4 loss_tf" [--dataset nominal]
#       [--cache_workers 16]
# Cluster wrapper: cluster/sail/train_vjepa2_pack.yaml (launcher: cluster/sail/sweep_vjepa2.py).
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict TRAIN_VJEPA2_PACK_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array TRAIN_VJEPA2_PACK_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

robot="${ARGS["robot"]}" dataset="${ARGS["dataset"]}"
( use_env vjepa2
  pids=()
  for i in $(seq 1 "${ARGS["cache_workers"]}"); do
      python -m pwm.wm.vjepa2.data build-cache --robot "$robot" --dataset "$dataset" --splits train,val &
      pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "[pack] frame cache worker failed"; exit 1; }; done
  python -m pwm.wm.vjepa2.data build-cache --robot "$robot" --dataset "$dataset" --splits train,val || exit 1
  pids=()
  for run in ${ARGS["runs"]}; do
      out="${storage_dir%/}/checkpoints/vjepa2/${robot}/${dataset}-${run}"
      mkdir -p "$out"
      WANDB_RUN_ID="vj2-${robot}-${run}" EXPERIMENT_TAG="vj2-${robot}-${run}" \
          python -m pwm.wm.vjepa2.train --robot "$robot" --dataset "$dataset" --run "$run" \
          --out "$out" > "$out/train.log" 2>&1 &
      pids+=($!)
      echo "[pack] $run -> $out (pid $!)"
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  exit $status )
