#!/usr/bin/env bash
# Builds the robocasa_gr1 environment (RoboCasa GR-1 tabletop sim client) into $PWM_ENV_DIR/robocasa_gr1
# (see setup/envs/README.md). Idempotent; --force rebuilds. Downloads ~3 GB of tabletop assets into the robocasa source.
#   bash setup/envs/robocasa_gr1/build.sh [--force <true|false>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
source setup/envs/common.sh || { echo "Could not source setup/envs/common.sh"; exit 1; }
declare -A ARGS
populate_dict BUILD_ENV_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array BUILD_ENV_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -euo pipefail

ENV_NAME=robocasa_gr1
ENV_SRC="setup/envs/${ENV_NAME}"
if [[ "${ARGS["force"]}" != "true" ]] && env_is_current "$ENV_NAME" "$ENV_SRC"; then
    echo "[skip] ${ENV_NAME} is up to date (${PWM_ENV_DIR}/${ENV_NAME}.tar)"; exit 0
fi
ROOT=$(env_build_root "$ENV_NAME")
fetch_sources "$ENV_SRC" "$ROOT/src"
sync_venv "$ENV_SRC" "$ROOT/venv"

# Tabletop assets (textures, fixtures, objaverse objects) live inside the robocasa package. RoboCasa walks ~80k of these
# files when it creates an env, which is why jobs unpack the environment to node-local disk.
RC="$ROOT/src/Isaac-GR00T/external_dependencies/robocasa-gr1-tabletop-tasks"
PYTHONPATH="$RC" "$ROOT/venv/bin/python" "$RC/robocasa/scripts/download_tabletop_assets.py" -y

PYTHONPATH="$RC:$ROOT/src/Isaac-GR00T" "$ROOT/venv/bin/python" -c "import robosuite, robocasa, gr00t; print('imports ok, robosuite', robosuite.__version__)"
finish_env "$ENV_NAME" "$ENV_SRC" "$ROOT"
