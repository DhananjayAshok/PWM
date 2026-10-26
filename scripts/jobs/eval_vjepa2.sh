#!/usr/bin/env bash
# V-JEPA 2-AC entry points of the eval contract (plans/predictions_format.md), one process per GPU:
#   --mode predict: predicted features for every episode of a split (needs --checkpoint, a best.pt)
#   --mode encode:  real-frame features of a split (nominal train sample, calibration pools, test sets)
#   bash scripts/jobs/eval_vjepa2.sh --robot <robot> --dataset <dataset> --mode <predict|encode> [--split test]
#       [--checkpoint <best.pt>] [--max_episodes 0]
# Cluster wrapper: cluster/sail/eval_vjepa2.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EVAL_VJEPA2_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EVAL_VJEPA2_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

robot="${ARGS["robot"]}" dataset="${ARGS["dataset"]}" split="${ARGS["split"]}"
gpus=$(nvidia-smi -L | wc -l)
( use_env vjepa2
  python -m pwm.wm.vjepa2.data build-cache --robot "$robot" --dataset "$dataset" --splits "$split" || exit 1
  pids=()
  for g in $(seq 0 $((gpus - 1))); do
      if [[ "${ARGS["mode"]}" == "predict" ]]; then
          [[ "${ARGS["checkpoint"]}" == "none" ]] && { echo "--checkpoint is required for --mode predict"; exit 1; }
          CUDA_VISIBLE_DEVICES=$g python -m pwm.wm.vjepa2.predict predict --robot "$robot" --dataset "$dataset" \
              --split "$split" --checkpoint "${ARGS["checkpoint"]}" --shard "$g/$gpus" &
      else
          CUDA_VISIBLE_DEVICES=$g python -m pwm.wm.vjepa2.predict encode --robot "$robot" --dataset "$dataset" \
              --split "$split" --max_episodes "${ARGS["max_episodes"]}" --shard "$g/$gpus" &
      fi
      pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  exit $status )
