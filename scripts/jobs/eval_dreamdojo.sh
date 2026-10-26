#!/usr/bin/env bash
# DreamDojo predictions (mode predict: first 4.8 s from frame 0, 1 context frame, free-running, seed 0) or real-frame
# VAE latents (mode encode, for the state probes) in the layout of plans/predictions_format.md, one process per GPU
# (pwm/wm/dreamdojo/predict.py). Finished episodes are skipped, so reruns resume.
#   bash scripts/jobs/eval_dreamdojo.sh --robot <robot> --run_dir <dir> [--dataset nominal] [--split test]
#   bash scripts/jobs/eval_dreamdojo.sh --robot <robot> --mode encode [--split train] [--max_episodes 2000]
# Cluster wrapper: cluster/sail/eval_dreamdojo.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EVAL_DREAMDOJO_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EVAL_DREAMDOJO_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

dataset="${storage_dir%/}/data/${ARGS["robot"]}/${ARGS["dataset"]}"
log_dir="${results_dir%/}/logs/eval_dreamdojo_${ARGS["robot"]}_${ARGS["dataset"]}_${ARGS["split"]}_${ARGS["mode"]}"
mkdir -p "$log_dir"
if [[ "${ARGS["mode"]}" == "predict" ]]; then
    run_dir="${ARGS["run_dir"]}"
    [[ "$run_dir" != "none" ]] || { echo "--run_dir is required for mode predict"; exit 1; }
    case "${ARGS["weights"]}" in
        best) weights="$run_dir/best_ema_bf16.pt" ;;
        latest) weights="$run_dir/ema_bf16_latest.pt" ;;
        *) weights="${ARGS["weights"]}" ;;
    esac
    tag="${ARGS["checkpoint_tag"]}"
    [[ "$tag" == "none" ]] && tag="$(basename "$run_dir")-${ARGS["weights"]}"
    cmd=(predict --dataset "$dataset" --split "${ARGS["split"]}" --weights "$weights" --stats "$run_dir/action_stats.json"
         --checkpoint_tag "$tag" --episodes "${ARGS["episodes"]}" --seed "${ARGS["seed"]}")
else
    tag="${ARGS["checkpoint_tag"]}"
    [[ "$tag" == "none" ]] && tag=vae
    cmd=(encode --dataset "$dataset" --split "${ARGS["split"]}" --checkpoint_tag "$tag" --max_episodes "${ARGS["max_episodes"]}")
fi
n=$(nvidia-smi -L | wc -l)
pids=()
for ((i = 0; i < n; i++)); do
    ( use_env dreamdojo
      CUDA_VISIBLE_DEVICES=$i exec python -m pwm.wm.dreamdojo.predict "${cmd[0]}" --robot "${ARGS["robot"]}" \
          "${cmd[@]:1}" --shard "$i/$n" ) > "$log_dir/gpu$i.log" 2>&1 &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "[eval_dreamdojo] ${ARGS["mode"]} ${ARGS["robot"]} ${ARGS["dataset"]}/${ARGS["split"]} done (status $status), logs in $log_dir"
exit $status
