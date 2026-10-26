#!/usr/bin/env bash
# Download and convert DreamDojo's released weights into <model_dir>/dreamdojo (pwm/wm/dreamdojo/assets.py). CPU only
# (~40 GB RAM for the conversion). Idempotent.
#   bash scripts/jobs/prepare_dreamdojo.sh
# Cluster wrapper: cluster/sail/prepare_dreamdojo.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict PREPARE_DREAMDOJO_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array PREPARE_DREAMDOJO_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -euo pipefail

( use_env dreamdojo
  python -m pwm.wm.dreamdojo.assets )
