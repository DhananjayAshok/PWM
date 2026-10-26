#!/usr/bin/env bash
# Builds the metrics environment (evaluation panel) into $PWM_ENV_DIR/metrics (see setup/envs/README.md).
# Idempotent; --force rebuilds.
#   bash setup/envs/metrics/build.sh [--force <true|false>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
source setup/envs/common.sh || { echo "Could not source setup/envs/common.sh"; exit 1; }
declare -A ARGS
populate_dict BUILD_ENV_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array BUILD_ENV_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -euo pipefail

ENV_NAME=metrics
ENV_SRC="setup/envs/${ENV_NAME}"
if [[ "${ARGS["force"]}" != "true" ]] && env_is_current "$ENV_NAME" "$ENV_SRC"; then
    echo "[skip] ${ENV_NAME} is up to date (${PWM_ENV_DIR}/${ENV_NAME}.tar)"; exit 0
fi
ROOT=$(env_build_root "$ENV_NAME")
sync_venv "$ENV_SRC" "$ROOT/venv"
finish_env "$ENV_NAME" "$ENV_SRC" "$ROOT"
