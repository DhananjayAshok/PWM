#!/usr/bin/env bash
# Step-1 acceptance for one robot: run its policy on its published tasks in our environments, then compare with the
# published success rates (pwm/verify.py -> <results_dir>/verification/<robot>.{json,md}). Needs one GPU; restart-safe
# (finished episodes are kept and skipped; GR-1 tasks rerun only if incomplete).
#   bash scripts/jobs/verify_robot.sh --robot <franka_libero|gr1_robocasa|widowx_bridge|google_fractal>
#       [--episodes <per task>] [--port 5555] [--n_envs 5]
# Cluster wrapper: cluster/sail/verify_robot.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict VERIFY_ROBOT_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array VERIFY_ROBOT_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

robot="${ARGS["robot"]}"
port="${ARGS["port"]}"
sim_env=$(python scripts/get_strings.py robot_field --robot "$robot" --field sim_env) || exit 1
policy_env=$(python scripts/get_strings.py robot_field --robot "$robot" --field policy_env) || exit 1
read -ra tasks <<< "$(python scripts/get_strings.py robot_tasks --robot "$robot")"
log_dir="${results_dir%/}/logs/verify_${robot}"
mkdir -p "$log_dir"
declare -A rollout_dirs
for task in "${tasks[@]}"; do
    rollout_dirs["$task"]=$(python scripts/get_strings.py verify_rollout_dir --robot "$robot" --task "$task") || exit 1
done
echo "[verify] $robot: sim_env=$sim_env policy_env=$policy_env, ${#tasks[@]} tasks, logs in $log_dir"

server_pid=""
function stop_server() {
    [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null
}
trap stop_server EXIT

if [[ "$robot" == "franka_libero" ]]; then
    episodes_flag=""
    [[ "${ARGS["episodes"]}" != "none" ]] && episodes_flag="--episodes_per_task ${ARGS["episodes"]}"
    for task in "${tasks[@]}"; do  # one GPU process: suites run one after another
        ( use_env "$sim_env"
          python -m pwm.sim.libero --robot "$robot" --suite "$task" --out "${rollout_dirs[$task]}" $episodes_flag ) \
            > "$log_dir/${task}.log" 2>&1 || echo "[verify] $task failed, see $log_dir/${task}.log"
    done
else
    ( use_env "$policy_env"; exec python -m pwm.policies.gr00t_server --robot "$robot" --port "$port" ) \
        > "$log_dir/server.log" 2>&1 &
    server_pid=$!
    until grep -q "Server is ready" "$log_dir/server.log"; do
        kill -0 "$server_pid" 2>/dev/null || { echo "[verify] policy server died:"; tail -30 "$log_dir/server.log"; exit 1; }
        sleep 10
    done
    echo "[verify] policy server ready"
    pids=()
    for task in "${tasks[@]}"; do  # all tasks in parallel against one server
        name=$(basename "${rollout_dirs[$task]}")
        if [[ "$robot" == "gr1_robocasa" ]]; then
            episodes="${ARGS["episodes"]}"; [[ "$episodes" == "none" ]] && episodes=20
            done_count=$(cat "${rollout_dirs[$task]}/episodes.jsonl" 2>/dev/null | wc -l)
            [[ "$done_count" -ge "$episodes" ]] && { echo "[verify] $name already has $done_count episodes"; continue; }
            ( use_env "$sim_env"
              python -m pwm.sim.robocasa_gr1 --robot "$robot" --task "$task" --out "${rollout_dirs[$task]}" \
                  --episodes "$episodes" --n_envs "${ARGS["n_envs"]}" --port "$port" ) > "$log_dir/${name}.log" 2>&1 &
        else
            episodes="${ARGS["episodes"]}"; [[ "$episodes" == "none" ]] && episodes=24
            ( use_env "$sim_env"
              python -m pwm.sim.simpler --robot "$robot" --task "$task" --out "${rollout_dirs[$task]}" \
                  --episodes "$episodes" --port "$port" ) > "$log_dir/${name}.log" 2>&1 &
        fi
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "[verify] a rollout client exited with an error (logs in $log_dir)"
    done
    stop_server
fi

python -m pwm.verify --robot "$robot"
