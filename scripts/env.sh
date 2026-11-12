#!/usr/bin/env bash
REPRO_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOCASA_ROOT="${ROBOCASA_ROOT:-/root/autodl-tmp/robocasa-repro}"
export HF_HOME="$ROBOCASA_ROOT/cache/huggingface"
export UV_CACHE_DIR="$ROBOCASA_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROBOCASA_ROOT/cache/python"
export XDG_CACHE_HOME="$ROBOCASA_ROOT/cache/xdg"
export OPENPI_DATA_HOME="$ROBOCASA_ROOT/models/openpi"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES="$REPRO_CODE_DIR/configs/10_nvidia.json"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_DISABLE_XET=1
export NO_PROXY=localhost,127.0.0.1,mirrors.aliyun.com,mirrors.tuna.tsinghua.edu.cn,repo.huaweicloud.com
export no_proxy="$NO_PROXY"
if [[ "${USE_LOCAL_PROXY:-1}" == 1 ]]; then
  export HTTP_PROXY="${REPRO_PROXY_URL:-http://127.0.0.1:17897}"
  export HTTPS_PROXY="$HTTP_PROXY" ALL_PROXY="$HTTP_PROXY"
  export http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY" all_proxy="$HTTP_PROXY"
fi
export PATH="/root/miniconda3/bin:$PATH"
export UV_HTTP_TIMEOUT=600
export REPRO_PYPI_INDEX="${REPRO_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}"

# Inference stays on the authenticated local computer; SSH forwards this port.
export CODEX_BRIDGE_UPSTREAM="${CODEX_BRIDGE_UPSTREAM:-http://127.0.0.1:18111}"
