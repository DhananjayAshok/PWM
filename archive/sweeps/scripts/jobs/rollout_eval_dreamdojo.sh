#!/usr/bin/env bash
# Rollout evaluation of a DreamDojo run (pwm/wm/dreamdojo/rollout_eval.py): 4.8 s free-running error on 32 fixed val
# episodes and action sensitivity, sharded over the visible GPUs. Results go to the run's status mirror
# (<results_dir>/sweeps/dreamdojo/runs/<run name>/rollout_eval/plain/), read by the sweep report.
# Finished episodes are skipped, so reruns resume.
#   bash scripts/jobs/rollout_eval_dreamdojo.sh --robot <robot> --run_dir <dir> [--weights best] [--dataset nominal]
# Cluster wrapper: cluster/sail/rollout_eval_dreamdojo.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict ROLLOUT_EVAL_DREAMDOJO_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array ROLLOUT_EVAL_DREAMDOJO_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

run_dir="${ARGS["run_dir"]%/}"
out="${results_dir%/}/sweeps/dreamdojo/runs/$(basename "$run_dir")/rollout_eval"
mkdir -p "$out"
n=$(nvidia-smi -L | wc -l)
status=0
pids=()
for ((i = 0; i < n; i++)); do
    ( use_env dreamdojo
      CUDA_VISIBLE_DEVICES=$i exec python -m pwm.wm.dreamdojo.rollout_eval --robot "${ARGS["robot"]}" \
          --run_dir "$run_dir" --weights "${ARGS["weights"]}" \
          --dataset "${storage_dir%/}/data/${ARGS["robot"]}/${ARGS["dataset"]}" --out "$out" --shard "$i/$n" ) \
        > "$out/plain_gpu$i.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "[rollout_eval] $(ls "$out/plain"/*.json 2>/dev/null | grep -vcE "eval_info.json|variant.json") episodes"
exit $status
