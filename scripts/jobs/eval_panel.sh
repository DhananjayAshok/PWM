#!/usr/bin/env bash
# Eval panel for one world model and robot: metrics on the given test sets (predictions must exist), then the tables.
#   bash scripts/jobs/eval_panel.sh --model <dreamdojo|vjepa2|dreamerv3> --robot <robot> --tag <checkpoint tag>
#       [--datasets nominal-test,shift-<s>-L<l>,... | all]
# Cluster wrapper: cluster/sail/eval_panel.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict EVAL_PANEL_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array EVAL_PANEL_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

datasets="${ARGS["datasets"]}"
if [[ "$datasets" == "all" ]]; then
    datasets=$(ls "${storage_dir%/}/data/${ARGS["robot"]}" | grep -E '^(nominal-test|shift-.*-L[1-5])$' | paste -sd, -)
fi
( use_env metrics
  python -m pwm.eval.panel --model "${ARGS["model"]}" --robot "${ARGS["robot"]}" --tag "${ARGS["tag"]}" --datasets "$datasets" ) \
    || exit 1
python -m pwm.eval.tables --tag "${ARGS["tag"]}"
