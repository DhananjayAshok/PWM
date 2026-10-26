#!/usr/bin/env bash
# Data generation on one GPU: N restart-safe workers (pwm/data/worker.py) for one robot and dataset, plus the GR00T
# policy server they share (robots whose policy runs in its own environment). Any number of these jobs can run at once
# on the same dataset; they coordinate through claim files. Exits when the dataset is complete.
#   bash scripts/jobs/generate.sh --robot <robot> [--kind nominal|test|calib] [--shift <name> --level <1-5>]
#       [--workers 8] [--groups_per_task none] [--calib_units 200] [--port 5555]
# Cluster wrapper: cluster/sail/generate.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict GENERATE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array GENERATE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

robot="${ARGS["robot"]}"
port="${ARGS["port"]}"
sim_env=$(python scripts/get_strings.py robot_field --robot "$robot" --field sim_env) || exit 1
policy_env=$(python scripts/get_strings.py robot_field --robot "$robot" --field policy_env) || exit 1
tag="${robot}_${ARGS["kind"]}_${ARGS["shift"]}_${ARGS["level"]}_$(date -u +%Y%m%dT%H%M%S)_$$"
log_dir="${results_dir%/}/logs/generate/${tag}"
mkdir -p "$log_dir"
worker_flags="--robot $robot --kind ${ARGS["kind"]} --port $port --calib_units ${ARGS["calib_units"]}"
[[ "${ARGS["shift"]}" != "none" ]] && worker_flags+=" --shift ${ARGS["shift"]} --level ${ARGS["level"]}"
[[ "${ARGS["groups_per_task"]}" != "none" ]] && worker_flags+=" --groups_per_task ${ARGS["groups_per_task"]}"

server_pid=""
function cleanup() { [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null; }
trap cleanup EXIT

# plan once (idempotent), so that workers starting together agree on the units
( use_env "$sim_env"; python -m pwm.data.worker $worker_flags --plan_only ) > "$log_dir/plan.log" 2>&1 \
    || { echo "[generate] planning failed:"; tail -30 "$log_dir/plan.log"; exit 1; }

if [[ "$policy_env" != "$sim_env" ]]; then
    ( use_env "$policy_env"; exec python -m pwm.policies.gr00t_server --robot "$robot" --port "$port" ) \
        > "$log_dir/server.log" 2>&1 &
    server_pid=$!
    until grep -q "Server is ready" "$log_dir/server.log"; do
        kill -0 "$server_pid" 2>/dev/null || { echo "[generate] policy server died:"; tail -30 "$log_dir/server.log"; exit 1; }
        sleep 10
    done
    echo "[generate] policy server ready"
fi

pids=()
for i in $(seq 1 "${ARGS["workers"]}"); do
    ( use_env "$sim_env"; exec python -m pwm.data.worker $worker_flags ) > "$log_dir/worker_${i}.log" 2>&1 &
    pids+=($!)
    sleep 5   # stagger env creation and policy loading
done
echo "[generate] ${#pids[@]} workers started for $robot ${ARGS["kind"]} (logs $log_dir)"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "[generate] workers done (status $status)"
exit $status
