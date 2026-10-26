# Runtime variables for the simpler environment; sourced by use_env (scripts/utils.sh) after activating its venv.
SIMPLER_SRC="${PWM_ACTIVE_ENV_ROOT}/src/Isaac-GR00T/external_dependencies/SimplerEnv"
export PYTHONPATH="${SIMPLER_SRC}:${SIMPLER_SRC}/ManiSkill2_real2sim${PYTHONPATH:+:$PYTHONPATH}"
export MS2_REAL2SIM_ASSET_DIR="${SIMPLER_SRC}/ManiSkill2_real2sim/data"
unset SIMPLER_SRC
# SAPIEN renders with Vulkan: use the NVIDIA ICD (cluster/sail/prepare_node.sh installs it if the image lacks one).
for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
    [[ -f "$icd" ]] && { export VK_ICD_FILENAMES="$icd"; break; }
done
unset icd
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
