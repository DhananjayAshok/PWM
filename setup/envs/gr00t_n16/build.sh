#!/usr/bin/env bash
# Builds the gr00t_n16 environment (GR00T N1.6 policy server) into $PWM_ENV_DIR/gr00t_n16 (see setup/envs/README.md).
# Idempotent; --force rebuilds. Model weights are not part of the environment (fetched into HF_HOME on first use).
#   bash setup/envs/gr00t_n16/build.sh [--force <true|false>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
source setup/envs/common.sh || { echo "Could not source setup/envs/common.sh"; exit 1; }
declare -A ARGS
populate_dict BUILD_ENV_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array BUILD_ENV_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -euo pipefail

ENV_NAME=gr00t_n16
ENV_SRC="setup/envs/${ENV_NAME}"
if [[ "${ARGS["force"]}" != "true" ]] && env_is_current "$ENV_NAME" "$ENV_SRC"; then
    echo "[skip] ${ENV_NAME} is up to date (${PWM_ENV_DIR}/${ENV_NAME}.tar)"; exit 0
fi
ROOT=$(env_build_root "$ENV_NAME")
fetch_sources "$ENV_SRC" "$ROOT/src"
sync_venv "$ENV_SRC" "$ROOT/venv"
finish_env "$ENV_NAME" "$ENV_SRC" "$ROOT"
