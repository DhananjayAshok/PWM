# Runtime variables for the libero environment; sourced by use_env (scripts/utils.sh) after activating its venv, and
# by build.sh.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# LIBERO reads its paths from a config file and asks for them interactively if it is missing. The environment can be
# unpacked anywhere (node-local disk in jobs), so the config is written here for the active location. The scene assets
# live inside the package, where LIBERO looks first.
LIBERO_PACKAGE="${PWM_ACTIVE_ENV_ROOT}/venv/lib/python3.12/site-packages/libero/libero"
export LIBERO_CONFIG_PATH="${PWM_ACTIVE_ENV_ROOT}/libero_config"
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: ${LIBERO_PACKAGE}
bddl_files: ${LIBERO_PACKAGE}/bddl_files
init_states: ${LIBERO_PACKAGE}/init_files
datasets: ${PWM_ACTIVE_ENV_ROOT}/datasets
assets: ${LIBERO_PACKAGE}/assets
YAML
unset LIBERO_PACKAGE
# Gated google/paligemma-3b-pt-224 tokenizer replaced by an identical one built from the public big_vision file.
export PWM_PI05_TOKENIZER="${PWM_ACTIVE_ENV_ROOT}/assets/paligemma_tokenizer"
# The container's torch libraries must not shadow the venv's own torch.
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i torch | paste -sd: -)
export LD_LIBRARY_PATH
# B300 (sm_103): cuDNN has no execution plans for some ops in cu128 wheels; policies run with cuDNN off.
export PWM_DISABLE_CUDNN=1
