# Runtime variables for the gr00t_n17 environment; sourced by use_env (scripts/utils.sh) after activating its venv.
export PYTHONPATH="${PWM_ACTIVE_ENV_ROOT}/src/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# The container's torch libraries must not shadow the venv's own torch.
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i torch | paste -sd: -)
export LD_LIBRARY_PATH
