#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBOCASA_ROOT"
python "$REPRO_CODE_DIR/scripts/configure_macros.py"
python "$REPRO_CODE_DIR/scripts/apply_patches.py"
if [[ ! -x envs/sim/bin/python || ! -x envs/policy/bin/python ]]; then uv python install 3.11; fi
for name in sim policy; do
  [[ -x "envs/$name/bin/python" ]] || uv venv --python 3.11 "envs/$name"
  uv pip install --python "envs/$name/bin/python" --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1+cpu torchvision==0.22.1+cpu
  uv pip install --python "envs/$name/bin/python" --index-url "$REPRO_PYPI_INDEX" \
    -e repos/robosuite -e repos/robocasa -e repos/openpi/packages/openpi-client
done
uv pip install --python envs/policy/bin/python --index-url "$REPRO_PYPI_INDEX" -e repos/openpi
uv pip install --python envs/policy/bin/python --index-url "$REPRO_PYPI_INDEX" chex==0.1.89
uv pip install --python envs/sim/bin/python --index-url "$REPRO_PYPI_INDEX" \
  -e repos/VoLoAgent fastapi uvicorn httpx imageio-ffmpeg
