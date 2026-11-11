#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBOCASA_ROOT/repos/openpi"
exec "$ROBOCASA_ROOT/envs/policy/bin/python" scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_pretrain_human300 \
  --policy.dir="$ROBOCASA_ROOT/models/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000"
