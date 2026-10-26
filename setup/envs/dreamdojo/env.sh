# Runtime variables for the dreamdojo environment; sourced by use_env (scripts/utils.sh) after activating its venv.
DD_SRC="${PWM_ACTIVE_ENV_ROOT}/src/DreamDojo"
export PWM_DREAMDOJO_SRC="$DD_SRC"
# DreamDojo's packages run from source; pwm/wm/dreamdojo/stubs stands in for pytorch3d, which groot_dreams imports but
# never calls.
export PYTHONPATH="${DD_SRC}:${DD_SRC}/packages/cosmos-oss:${DD_SRC}/packages/cosmos-cuda:${PROJECT_ROOT}/pwm/wm/dreamdojo/stubs${PYTHONPATH:+:$PYTHONPATH}"
unset DD_SRC
export PYTHONUNBUFFERED=1
# The container's torch libraries must not shadow the venv's own torch.
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i torch | paste -sd: -)
export LD_LIBRARY_PATH
