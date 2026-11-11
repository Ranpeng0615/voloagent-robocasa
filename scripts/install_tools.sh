#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBOCASA_ROOT"
# Keep GPU tool dependencies separate from π's JAX and the CPU simulation env.
[[ -x envs/tools/bin/python ]] || uv venv --system-site-packages --python /root/miniconda3/bin/python envs/tools
uv pip install --python envs/tools/bin/python --index-url "$REPRO_PYPI_INDEX" \
  'numpy==1.26.4' omegaconf hydra-core scipy h5py trimesh meshcat scikit-learn \
  einops 'timm>=1.0.17' ftfy iopath regex pillow 'huggingface_hub<1' webdataset \
  matplotlib tensorboard tensorboardx addict 'tensordict<0.9' diffusers fastapi uvicorn \
  opencv-python-headless requests yourdfpy setuptools wheel ninja cmake pycocotools \
  decord scikit-image 'transformers<5' cumm-cu128
uv pip install --python envs/tools/bin/python --no-deps --index-url "$REPRO_PYPI_INDEX" spconv-cu126
uv pip install --python envs/tools/bin/python --no-deps -e repos/GraspGen -e repos/sam3 -e repos/VoLoAgent
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
export CXX=/usr/bin/g++ CC=/usr/bin/gcc
export TORCH_EXTENSIONS_DIR="$ROBOCASA_ROOT/cache/torch_extensions"
export FORCE_ONLY_CUDA=1
uv pip install --python envs/tools/bin/python --index-url "$REPRO_PYPI_INDEX" \
  --no-build-isolation --no-deps torch-scatter
uv pip install --python envs/tools/bin/python --no-build-isolation --no-deps repos/GraspGen/pointnet2_ops
