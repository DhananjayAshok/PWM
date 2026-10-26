# Runtime variables for the vjepa2 environment; sourced by use_env (scripts/utils.sh) after activating its venv.
# vjepa2's code imports itself as `src.*` and `app.*`, so its repository root goes on PYTHONPATH.
export PYTHONPATH="${PWM_ACTIVE_ENV_ROOT}/src/vjepa2${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# torch.hub downloads (V-JEPA 2-AC checkpoint) go to the shared model directory, not ~/.cache.
export TORCH_HOME="${storage_dir%/}/models/torch_hub"
# The container's torch libraries must not shadow the venv's own torch.
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i torch | paste -sd: -)
export LD_LIBRARY_PATH
