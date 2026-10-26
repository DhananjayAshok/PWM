#!/usr/bin/env bash
# Node preparation for jobs on the Sail cluster (nvcr.io/nvidia/pytorch:26.05-py3 image), run at the top of every
# cluster/sail/*.yaml. The image has no OpenGL/EGL/Vulkan user-space libraries; the simulators need them:
# MuJoCo (LIBERO, RoboCasa) renders through NVIDIA EGL, SAPIEN (SimplerEnv) through Vulkan.
set -euo pipefail
SUDO=$([[ "$(id -u)" -eq 0 ]] || echo sudo)
$SUDO apt-get update -qq
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends \
    libgl1 libegl1 libvulkan1 libglib2.0-0 > /dev/null
if [[ ! -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    $SUDO mkdir -p /usr/share/glvnd/egl_vendor.d
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        | $SUDO tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json > /dev/null
fi
if [[ ! -e /usr/share/vulkan/icd.d/nvidia_icd.json && ! -e /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    $SUDO mkdir -p /usr/share/vulkan/icd.d
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
        | $SUDO tee /usr/share/vulkan/icd.d/nvidia_icd.json > /dev/null
fi
echo "[prepare_node] EGL + Vulkan ready"
