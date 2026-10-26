#!/usr/bin/env bash
# Submit a cluster/sail/*.yaml job as a SkyPilot managed job. Fills the job file's ${cluster_user} / ${cluster_volume}
# from cluster/sail/sail_vars.yaml (gitignored), passes this shell's HF_HOME, and always detaches.
#   bash cluster/sail/launch.sh <job.yaml> <job name> [--env KEY=VALUE ...] [other `sky jobs launch` flags]
# Run from the project root (the job gets a copy of it, see .skyignore).
set -euo pipefail
vars=cluster/sail/sail_vars.yaml
[[ -f "$vars" ]] || { echo "$vars not found; copy cluster/sail/sail_vars.example.yaml and fill it in"; exit 1; }
[[ -f configs/config.env ]] || { echo "configs/config.env not found; run: bash setup/setup.sh"; exit 1; }
[[ $# -ge 2 ]] || { echo "Usage: $0 <job.yaml> <job name> [sky jobs launch flags]"; exit 1; }
job_file="$1" job_name="$2"
shift 2
[[ -n "${HF_HOME:-}" ]] || { echo "HF_HOME is not set"; exit 1; }

function sail_var() {  # flat "key: value  # comment" lines
    sed -n "s/^$1: *\"\{0,1\}\([^\"#]*\)\"\{0,1\}.*/\1/p" "$vars" | sed 's/ *$//'
}
cluster_user=$(sail_var cluster_user)
cluster_volume=$(sail_var cluster_volume)
for v in cluster_user cluster_volume; do
    [[ -n "${!v}" && "${!v}" != "PLACEHOLDER" ]] || { echo "Set $v in $vars"; exit 1; }
done
rendered="cluster/sail/.rendered_$(basename "$job_file")"
sed -e "s|\${cluster_user}|${cluster_user}|g" -e "s|\${cluster_volume}|${cluster_volume}|g" "$job_file" > "$rendered"
sky jobs launch -d -y -n "$job_name" "$rendered" --env HF_HOME="$HF_HOME" "$@"
