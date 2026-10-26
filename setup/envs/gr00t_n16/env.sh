# Runtime variables for the gr00t_n16 environment; sourced by use_env (scripts/utils.sh) after activating its venv.
export PYTHONPATH="${PWM_ACTIVE_ENV_ROOT}/src/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
# The server prints "Server is ready" only when unbuffered (callers wait for that line).
export PYTHONUNBUFFERED=1
# Eagle backbone: SDPA instead of flash-attention 2 (patches/Isaac-GR00T/01_eagle_sdpa_fallback.diff).
export GR00T_ATTN=sdpa
# The container's torch libraries must not shadow the venv's own torch.
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i torch | paste -sd: -)
export LD_LIBRARY_PATH
