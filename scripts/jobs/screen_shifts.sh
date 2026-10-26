#!/usr/bin/env bash
# Shift-level screening on franka_libero (pwm/shifts/screen.py): pi0.5 on fixed train starts at every candidate value of
# every shift in configs/shifts.yaml. Restart-safe; shard with --worker / --num_workers. Needs one GPU.
#   bash scripts/jobs/screen_shifts.sh [--worker 0 --num_workers 4] [--shifts all] [--out none]
#   bash scripts/jobs/screen_shifts.sh --selftest_robot <robot> [--out <dir>]     (shift self-test on any robot)
# Cluster wrapper: cluster/sail/screen_shifts.yaml. Summary: python -m pwm.shifts.screen summarize --out <out>
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict SCREEN_SHIFTS_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array SCREEN_SHIFTS_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

out="${ARGS["out"]}"
[[ "$out" == "none" ]] && out="${storage_dir%/}/data/shift_screen/franka_libero"
mkdir -p "$out"
robot="${ARGS["selftest_robot"]}"
if [[ "$robot" != "none" ]]; then  # every shift at its strongest candidate on one robot (pwm/shifts/selftest.py)
    sim_env=$(python scripts/get_strings.py robot_field --robot "$robot" --field sim_env) || exit 1
    ( use_env "$sim_env"; python -m pwm.shifts.selftest --robot "$robot" --out "$out" --shifts "${ARGS["shifts"]}" )
    exit $?
fi
( use_env libero
  python -m pwm.shifts.screen run --out "$out" --worker "${ARGS["worker"]}" --num_workers "${ARGS["num_workers"]}" \
      --shifts "${ARGS["shifts"]}" --candidates_key "${ARGS["candidates_key"]}" \
      $([[ "${ARGS["confirm"]}" != "none" ]] && echo --confirm "${ARGS["confirm"]}") )
