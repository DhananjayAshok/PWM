# Runtime variables for the robocasa_gr1 environment; sourced by use_env (scripts/utils.sh) after activating its venv.
GR00T_SRC="${PWM_ACTIVE_ENV_ROOT}/src/Isaac-GR00T"
export PYTHONPATH="${GR00T_SRC}/external_dependencies/robocasa-gr1-tabletop-tasks:${GR00T_SRC}${PYTHONPATH:+:$PYTHONPATH}"
unset GR00T_SRC
# NVIDIA EGL in jobs (cluster/sail/prepare_node.sh); osmesa segfaults robosuite's numba kernels.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# Many clients per node: one thread each.
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i torch | paste -sd: -)
export LD_LIBRARY_PATH
