#!/usr/bin/env bash
# Fine-tune V-JEPA 2-AC's predictor on one robot's data until its validation loss stops improving
# (pwm/wm/vjepa2/train.py). Builds the dataset's frame cache first (parallel, restart-safe), then trains on every GPU of
# the node. Restart-safe: rerunning resumes from the newest checkpoint; a finished run (STOPPED.json) exits at once.
#   bash scripts/jobs/train_vjepa2.sh --robot <robot> [--dataset nominal] [--ckpt_dir <dir>] [--cache_workers 16]
# Settings: configs/world_models.yaml vjepa2. Checkpoints default to <storage_dir>/checkpoints/vjepa2/<robot>/<dataset>/.
# Cluster wrapper: cluster/sail/train_vjepa2.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict TRAIN_VJEPA2_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array TRAIN_VJEPA2_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

robot="${ARGS["robot"]}" dataset="${ARGS["dataset"]}"
ckpt_dir="${ARGS["ckpt_dir"]}"
[[ "$ckpt_dir" == "none" ]] && ckpt_dir="${storage_dir%/}/checkpoints/vjepa2/${robot}/${dataset}"
gpus=$(nvidia-smi -L | wc -l)
( use_env vjepa2
  pids=()
  for i in $(seq 1 "${ARGS["cache_workers"]}"); do
      python -m pwm.wm.vjepa2.data build-cache --robot "$robot" --dataset "$dataset" --splits train,val &
      pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "[train_vjepa2] frame cache worker failed"; exit 1; }; done
  # one more pass picks up episodes whose claims belonged to workers that died
  python -m pwm.wm.vjepa2.data build-cache --robot "$robot" --dataset "$dataset" --splits train,val || exit 1
  export MASTER_ADDR=127.0.0.1 MASTER_PORT=29511
  exec python -m torch.distributed.run --standalone --nproc_per_node="$gpus" -m pwm.wm.vjepa2.train \
      --robot "$robot" --dataset "$dataset" --out "$ckpt_dir" )
