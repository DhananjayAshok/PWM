#!/usr/bin/env bash
# Predictions of a trained DreamerV3 world model in the plans/predictions_format.md layout (skips finished episodes):
#   --mode predict: the fixed 4.8 s window from 1 real frame, free-running (pwm/wm/dreamerv3/predict.py)
#   --mode encode:  posterior latent states of real frames, for training the state probe (pwm/wm/dreamerv3/encode.py)
#   bash scripts/jobs/eval_dreamerv3.sh --robot <robot> [--mode predict|encode] [--model_dir ...] [--dataset nominal]
#       [--split test|train|calib] [--checkpoint_tag <tag>] [--samples 1] [--max_episodes N]
# Cluster wrapper: cluster/sail/eval_dreamerv3.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EVAL_DREAMERV3_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EVAL_DREAMERV3_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

robot="${ARGS["robot"]}"
mode="${ARGS["mode"]}"
model="${ARGS["model_dir"]}"
[[ "$model" == "none" ]] && model="${storage_dir%/}/models/dreamerv3/dv3-${robot}-nominal"
split="${ARGS["split"]}"
if [[ "$split" == "none" ]]; then
    [[ "$mode" == "encode" ]] && split=train || split=test
fi
tag_flag=""
[[ "${ARGS["checkpoint_tag"]}" != "none" ]] && tag_flag="--checkpoint_tag ${ARGS["checkpoint_tag"]}"
context=$(python -c "import yaml; print(yaml.safe_load(open('configs/world_models.yaml'))['dreamerv3']['context_frames'])")
case "$mode" in
    predict) cmd=(python -m pwm.wm.dreamerv3.predict --context "$context" --samples "${ARGS["samples"]}") ;;
    encode)  cmd=(python -m pwm.wm.dreamerv3.encode)
             [[ "${ARGS["max_episodes"]}" != "none" ]] && cmd+=(--max_episodes "${ARGS["max_episodes"]}") ;;
    *) echo "Unknown --mode $mode (predict | encode)"; exit 1 ;;
esac
( use_env dreamerv3
  exec "${cmd[@]}" --model_dir "$model" --robot "$robot" --dataset "${ARGS["dataset"]}" --split "$split" $tag_flag )
