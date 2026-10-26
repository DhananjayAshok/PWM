# Runtime variables for the metrics environment; sourced by use_env (scripts/utils.sh) after activating its venv.
# LPIPS downloads its VGG/AlexNet weights through torch.hub on first use: keep them under storage_dir.
export TORCH_HOME="${storage_dir%/}/cache/torch"
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v -i torch | paste -sd: -)
export LD_LIBRARY_PATH
