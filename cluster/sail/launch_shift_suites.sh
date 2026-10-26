#!/usr/bin/env bash
# Submit one generate_shift_suite job (all 5 levels, test + calibration) per robot x supported shift. Skips a robot x
# shift whose job is already queued or running. Run from the project root.
#   bash cluster/sail/launch_shift_suites.sh [--priority p2] [--robots "franka_libero ..."]
source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }
priority=p2
robots="franka_libero gr1_robocasa widowx_bridge google_fractal"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --priority) priority="$2"; shift 2 ;;
        --robots) robots="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
# counterfactual groups per task: roughly 12-40 groups (60-200 test episodes) per level
declare -A GROUPS_FOR=( [franka_libero]=1 [gr1_robocasa]=1 [widowx_bridge]=3 [google_fractal]=2 )
queued=$(sky jobs queue --skip-finished -a 2>/dev/null | awk '{print $3}')
for robot in $robots; do
    shifts=$(PYTHONPATH=$PWD python - "$robot" <<'PY'
import sys
from pwm.utils import load_parameters
from pwm.shifts import registry as R
p = load_parameters()
print(" ".join(n for n in R.shift_names(parameters=p) if not R.unsupported_reason(robot=sys.argv[1], name=n, parameters=p)))
PY
)
    for shift in $shifts; do
        name="pwm-gensh-${robot}-${shift}"
        if echo "$queued" | grep -qx "$name"; then echo "[skip] $name already queued"; continue; fi
        bash cluster/sail/launch.sh cluster/sail/generate_shift_suite.yaml "$name" --priority "$priority" \
            --env ROBOT="$robot" --env SHIFT="$shift" --env GROUPS_PER_TASK="${GROUPS_FOR[$robot]}" 2>&1 \
            | grep -E "Managed Job ID|rror" | sed "s/^/$name: /"
    done
done
