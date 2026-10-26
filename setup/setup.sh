#!/usr/bin/env bash
# One-time, idempotent project setup. Run from the project root after filling configs/private_vars.yaml:
#   bash setup/setup.sh [all | env ...]
# 1. core venv (setup/.venv) from setup/uv.lock
# 2. configs/config.env from configs/*.yaml
# 3. the named simulator/policy environments (all = every setup/envs/*/), each skipped if already up to date;
#    none if no argument (build them here or as a cluster job: cluster/sail/build_envs.yaml).
# GPU acceptance checks are separate: scripts/jobs/verify_robot.sh (cluster wrapper: cluster/sail/verify_robot.yaml).
set -euo pipefail
[[ -f setup/pyproject.toml ]] || { echo "Run from the project root"; exit 1; }
command -v uv > /dev/null || { echo "uv not found; install it with: pip install uv"; exit 1; }

# storage_dir is needed before the core venv (and its yaml parser) exists: uv's cache lives under it.
storage_dir=$(sed -n 's/^storage_dir: *"\{0,1\}\([^"#]*\)"\{0,1\}.*/\1/p' configs/private_vars.yaml | tr -d ' ')
if [[ -z "$storage_dir" || "$storage_dir" == "PLACEHOLDER" || "$storage_dir" != /* ]]; then
    echo "Set storage_dir (and results_dir) in configs/private_vars.yaml to absolute paths first."; exit 1
fi
export UV_CACHE_DIR="${storage_dir%/}/uv/cache" UV_PYTHON_INSTALL_DIR="${storage_dir%/}/uv/python" UV_LINK_MODE=copy

echo "[setup] core environment"
uv sync --frozen --project setup
setup/.venv/bin/python configs/create_env_file.py

envs=("$@")
if [[ "${1:-}" == "all" ]]; then
    envs=()
    for d in setup/envs/*/; do envs+=("$(basename "$d")"); done
fi
for env_name in "${envs[@]}"; do
    echo "[setup] environment ${env_name}"
    bash "setup/envs/${env_name}/build.sh"
done
echo "[setup] done. Environments are built into ${storage_dir%/}/envs"
