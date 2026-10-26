#!/usr/bin/env bash
# Build several environments (setup/envs/<env>/build.sh), each skipped if already up to date.
#   bash scripts/jobs/build_envs.sh [--envs "<env> <env> ..." | all] [--force <true|false>]
# Cluster wrapper: cluster/sail/build_envs.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict BUILD_ENVS_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array BUILD_ENVS_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"

if [[ "${ARGS["envs"]}" == "all" ]]; then
    envs=()
    for d in setup/envs/*/; do envs+=("$(basename "$d")"); done
else
    read -ra envs <<< "${ARGS["envs"]}"
fi
failed=()
for env_name in "${envs[@]}"; do
    echo "[build_envs] ${env_name}"
    bash "setup/envs/${env_name}/build.sh" --force "${ARGS["force"]}" || failed+=("$env_name")
done
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "[build_envs] FAILED: ${failed[*]}"; exit 1
fi
echo "[build_envs] all built: ${envs[*]}"
