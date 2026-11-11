#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export VOLO_SAM3_DEVICE=cuda
export VOLO_SAM3_CHECKPOINT="$ROBOCASA_ROOT/models/sam3/sam3.pt"
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR="$ROBOCASA_ROOT/cache/torch_extensions"
export PYTORCH_KERNEL_CACHE_PATH="$ROBOCASA_ROOT/cache/torch/kernels"
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
cd "$ROBOCASA_ROOT/repos/VoLoAgent"
exec "$ROBOCASA_ROOT/envs/tools/bin/python" -m vlm_orchestrator.grasp.server \
  --host 127.0.0.1 --port 8003 --enable-sam3 \
  --gripper-config "$ROBOCASA_ROOT/models/GraspGenModels/checkpoints/graspgen_franka_panda.yml"
