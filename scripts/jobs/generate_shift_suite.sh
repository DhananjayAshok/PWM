#!/usr/bin/env bash
# One robot x one shift: generate the shift test set (counterfactual groups) and the calibration pool for every level
# L1..L5 (configs/shifts.yaml), one after another, each with scripts/jobs/generate.sh. Restart-safe: finished units are
# skipped, so a re-run after eviction continues where it stopped.
#   bash scripts/jobs/generate_shift_suite.sh --robot <robot> --shift <name> [--levels "1 2 3 4 5"]
#       [--groups_per_task 10] [--calib_units 50] [--workers 8]
# Cluster wrapper: cluster/sail/generate_shift_suite.yaml.
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
declare -A ARGS
populate_dict GENERATE_SHIFT_SUITE_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array GENERATE_SHIFT_SUITE_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -uo pipefail

status=0
for level in ${ARGS["levels"]}; do
    for kind in test calib; do
        echo "[shift_suite] ${ARGS["robot"]} ${ARGS["shift"]} L${level} ${kind}"
        bash scripts/jobs/generate.sh --robot "${ARGS["robot"]}" --kind "$kind" --shift "${ARGS["shift"]}" \
            --level "$level" --workers "${ARGS["workers"]}" --groups_per_task "${ARGS["groups_per_task"]}" \
            --calib_units "${ARGS["calib_units"]}" || { echo "[shift_suite] L${level} ${kind} failed"; status=1; }
    done
done
echo "[shift_suite] done (status $status)"
exit $status
