#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROBOCASA_ROOT/repos/VoLoAgent"
# This selects VoLoAgent's MuJoCo observation profile. It does not install LIBERO.
# First episode uses instruction replanning; optional geometric tools are separate.
exec "$ROBOCASA_ROOT/envs/sim/bin/vlm-orchestrator" \
  --env libero --host 127.0.0.1 --port 8001 --vla-port 8000 --mode subgoal \
  --failure-monitor vlm --recovery-mode replan \
  --vlm-model gpt-6.1-sol --vlm-base-url http://127.0.0.1:8111/v1 \
  --vlm-api-key local-codex --log-dir "$RUN_DIR/orchestrator" --verbose
