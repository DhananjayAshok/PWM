# Runtime variables for the dreamerv3 environment; sourced by use_env (scripts/utils.sh) after activating its venv.
# The port is a flat set of modules (networks, models, tools) imported by name.
export PYTHONPATH="${PWM_ACTIVE_ENV_ROOT}/src/dreamerv3-torch${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# LPIPS backbone weights (torchvision download), cached under storage_dir
export TORCH_HOME="${storage_dir%/}/cache/torch"
# The container's torch libraries must not shadow the venv's own torch.
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i torch | paste -sd: -)
export LD_LIBRARY_PATH
