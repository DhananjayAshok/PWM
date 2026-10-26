#!/usr/bin/env bash
# `setup:` step of every cluster/sail/*.yaml job: node libraries, then the core venv from setup/uv.lock (the job's copy
# of the repo has no setup/.venv; see .skyignore). Run from the job's workdir (the project root).
set -euo pipefail
bash cluster/sail/prepare_node.sh
source configs/config.env
command -v uv > /dev/null || pip install -q uv
export UV_CACHE_DIR="${PWM_UV_CACHE_DIR:-${storage_dir%/}/uv/cache}" UV_PYTHON_INSTALL_DIR="${storage_dir%/}/uv/python" UV_LINK_MODE=copy
uv sync --frozen --project setup
echo "[job_setup] core environment ready"
